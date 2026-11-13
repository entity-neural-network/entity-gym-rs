use proc_macro2::TokenStream;
use quote::quote;
use syn::spanned::Spanned;
use syn::{
    parse_macro_input, Data, DeriveInput, Expr, ExprLit, Field, Ident, Lit, Type, TypeArray,
    TypePath,
};

pub fn derive(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    // Parse the input tokens into a syntax tree
    let input = parse_macro_input!(input as DeriveInput);

    let name = &input.ident;
    let (impl_generics, ty_generics, where_clause) = input.generics.split_for_impl();

    let (num_feats, features_names, featurize) = match field_names(&input.data, name) {
        Ok(generated) => generated,
        Err(err) => return err.to_compile_error().into(),
    };
    let name_str = name.to_string();

    // Build the output, possibly using quasi-quotation
    let expanded = quote! {
        #[allow(unused_qualifications)]
        impl #impl_generics Featurizable for #name #ty_generics #where_clause {
            fn num_feats() -> usize {
                #num_feats
            }

            fn feature_names() -> Vec<String> {
                let mut names = Vec::with_capacity(#num_feats);
                #features_names;
                names
            }

            fn featurize(&self) -> Vec<f32> {
                let mut buffer = Vec::with_capacity(#num_feats);
                #featurize;
                buffer
            }

            fn name() -> &'static str {
                #name_str
            }
        }
    };

    // Hand the output tokens back to the compiler
    proc_macro::TokenStream::from(expanded)
}

fn field_names(data: &Data, name: &Ident) -> syn::Result<(TokenStream, TokenStream, TokenStream)> {
    match data {
        Data::Struct(data) => match &data.fields {
            syn::Fields::Named(fields) => {
                let mut names = vec![];
                let mut counts = vec![];
                let mut features = vec![];
                for field in fields.named.iter() {
                    let (name, count) = feature_name_field(field)?;
                    names.push(name);
                    counts.push(count);
                    features.push(featurize_field(field)?)
                }

                Ok((
                    quote! { #(#counts)+* },
                    quote! { #(#names)* },
                    quote! { #(#features);* },
                ))
            }
            syn::Fields::Unnamed(fields) => Err(syn::Error::new(
                fields.span(),
                "Featurizable cannot be derived for tuple structs",
            )),
            syn::Fields::Unit => Ok((quote! { 0 }, quote! { (); }, quote! { (); })),
        },
        Data::Enum(data) => {
            let mut names = vec![];
            let count = data.variants.len();
            let mut features = vec![];
            for variant in &data.variants {
                if !variant.fields.is_empty() {
                    return Err(syn::Error::new(
                        variant.span(),
                        "Enum variants with fields are not supported",
                    ));
                }
                let field_name = format!("is_{}", variant.ident);
                names.push(quote! { names.push(#field_name.to_string()); });
                let variant_ident = &variant.ident;
                features.push(quote! {
                    buffer.push(if let #name::#variant_ident = self {
                        1.0
                    } else {
                        0.0
                    })
                })
            }

            Ok((
                quote! { #count },
                quote! { #(#names)* },
                quote! { #(#features);* },
            ))
        }
        Data::Union(data) => Err(syn::Error::new(
            data.union_token.span(),
            "Featurizable cannot be derived for unions",
        )),
    }
}

fn featurize_field(field: &Field) -> syn::Result<TokenStream> {
    let ident = field.ident.as_ref().unwrap();
    featurize_type(&field.ty, quote! { self.#ident })
}

fn featurize_type(ty: &Type, accessor: TokenStream) -> syn::Result<TokenStream> {
    match ty {
        Type::Path(ty) => {
            let ty = ty.path.segments.last().unwrap();
            Ok(match ty.ident.to_string().as_str() {
                "f32" => quote! { buffer.push(#accessor) },
                "bool" => quote! { buffer.push(if #accessor { 1.0 } else { 0.0 }) },
                "f64" | "u8" | "u16" | "u32" | "u64" | "i8" | "i16" | "i32" | "i64" | "usize"
                | "isize" => {
                    quote! { buffer.push(#accessor as f32) }
                }
                _ => quote!(buffer.extend(#accessor.featurize())),
            })
        }
        Type::Array(ty) => {
            let mut elems = vec![];
            for i in 0..array_len(ty)? {
                elems.push(featurize_type(&ty.elem, quote! { #accessor[#i] })?);
            }
            Ok(quote! {
                #(#elems);*;
            })
        }
        t => Err(unsupported_type(t)),
    }
}

fn is_primitive_type(ty: &TypePath) -> bool {
    let ty = ty.path.segments.last().unwrap();
    matches!(
        ty.ident.to_string().as_str(),
        "f32"
            | "f64"
            | "u8"
            | "u16"
            | "u32"
            | "u64"
            | "i8"
            | "i16"
            | "i32"
            | "i64"
            | "bool"
            | "usize"
            | "isize"
    )
}

fn feature_name_field(field: &Field) -> syn::Result<(TokenStream, TokenStream)> {
    let ident = field.ident.as_ref().unwrap().to_string();
    feature_name_type(&field.ty, &ident)
}

fn feature_name_type(ty: &Type, prefix: &str) -> syn::Result<(TokenStream, TokenStream)> {
    match ty {
        Type::Path(ty) if is_primitive_type(ty) => Ok((
            quote! { names.push(#prefix.to_string()); },
            quote! { 1usize },
        )),
        Type::Path(ty) => {
            let type_ident = &ty.path.segments.last().unwrap().ident;
            Ok((
                quote! {
                    for name in #type_ident::feature_names() {
                        names.push(format!("{}.{}", #prefix, name));
                    }
                },
                quote! { #type_ident::num_feats() },
            ))
        }
        Type::Array(ty) => {
            let mut names = vec![];
            let mut counts = vec![];
            for i in 0..array_len(ty)? {
                let (name, count) = feature_name_type(&ty.elem, &format!("{}.{}", prefix, i))?;
                names.push(name);
                counts.push(count);
            }
            Ok((quote! { #(#names)* }, quote! { #(#counts)+* }))
        }
        t => Err(unsupported_type(t)),
    }
}

/// The length of an array type, which must be an integer literal.
fn array_len(ty: &TypeArray) -> syn::Result<usize> {
    match &ty.len {
        Expr::Lit(ExprLit {
            lit: Lit::Int(int), ..
        }) => int
            .base10_parse::<usize>()
            .map_err(|_| syn::Error::new(int.span(), "Array length must be a valid integer")),
        x => Err(syn::Error::new(
            x.span(),
            "Array length must be an integer literal",
        )),
    }
}

fn unsupported_type(t: &Type) -> syn::Error {
    syn::Error::new_spanned(
        t,
        format!(
            "Can't derive Featurizable for type `{}`",
            quote!(#t).to_string()
        ),
    )
}
