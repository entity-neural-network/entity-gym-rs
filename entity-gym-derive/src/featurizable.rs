use proc_macro2::TokenStream;
use proc_macro_error::abort;
use quote::quote;
use syn::spanned::Spanned;
use syn::{parse_macro_input, Data, DeriveInput, Field, Ident, Type, TypePath};

pub fn derive(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    // Parse the input tokens into a syntax tree
    let input = parse_macro_input!(input as DeriveInput);

    let name = &input.ident;
    let (impl_generics, ty_generics, where_clause) = input.generics.split_for_impl();

    let (num_feats, features_names, featurize) = field_names(&input.data, name);
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

fn field_names(data: &Data, name: &Ident) -> (TokenStream, TokenStream, TokenStream) {
    match data {
        Data::Struct(data) => match &data.fields {
            syn::Fields::Named(fields) => {
                let mut names = vec![];
                let count = fields.named.len();
                let mut features = vec![];
                for field in fields.named.iter() {
                    names.push(featurize_name(field));
                    features.push(featurize_field(field))
                }

                (
                    quote! { #count },
                    quote! { #(#names)* },
                    quote! { #(#features);* },
                )
            }
            syn::Fields::Unnamed(_fields) => todo!(),
            // syn::Fields::Unnamed(fields) => {
            //     let mut names = vec![];
            //     for i in 0..fields.unnamed.len() {
            //         let name = format!("_{}", i);
            //         names.push(quote! { #name });
            //     }
            //     let count = fields.unnamed.len();
            //     (quote! { #count }, quote! { &[#(#names),*] })
            // }
            syn::Fields::Unit => (quote! { 0 }, quote! { (); }, quote! { (); }),
        },
        Data::Enum(data) => {
            let mut names = vec![];
            let count = data.variants.len();
            let mut features = vec![];
            for variant in &data.variants {
                if !variant.fields.is_empty() {
                    abort!(
                        variant.span(),
                        "Enum variants with fields are not supported",
                    );
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

            (
                quote! { #count },
                quote! { #(#names)* },
                quote! { #(#features);* },
            )
        }
        Data::Union(_) => unimplemented!("Union not supported, must be struct"),
    }
}

fn featurize_field(field: &Field) -> TokenStream {
    let ident = field.ident.as_ref().unwrap();
    match &field.ty {
        Type::Path(ty) => {
            let ty = ty.path.segments.last().unwrap();
            match ty.ident.to_string().as_str() {
                "f32" => quote! { buffer.push(self.#ident) },
                "f64" | "u8" | "u16" | "u32" | "u64" | "i8" | "i16" | "i32" | "i64" => {
                    quote! { buffer.push(self.#ident as f32) }
                }
                "bool" => quote! { buffer.push(if self.#ident { 1.0 } else { 0.0 }) },
                _ => quote!(buffer.extend(self.#ident.featurize())),
            }
        }
        t => abort!(
            t,
            "Can't derive Featurizable for type `{}`",
            quote!(#t).to_string(),
        ),
    }
}

fn is_primitive_type(ty: &TypePath) -> bool {
    let ty = ty.path.segments.last().unwrap();
    match ty.ident.to_string().as_str() {
        "f32" | "f64" | "u8" | "u16" | "u32" | "u64" | "i8" | "i16" | "i32" | "i64" | "bool" => {
            true
        }
        _ => false,
    }
}

fn featurize_name(field: &Field) -> TokenStream {
    let ident = field.ident.as_ref().unwrap().to_string();
    match &field.ty {
        Type::Path(ty) => {
            if is_primitive_type(ty) {
                quote! { names.push(#ident.to_string()); }
            } else {
                let type_ident = &ty.path.segments.last().unwrap().ident;
                quote! {
                    for name in #type_ident::feature_names() {
                        names.push(format!("{}.{}", #ident, name));
                    }
                }
            }
        }
        t => abort!(
            t,
            "Can't derive Featurizable for type `{}`",
            quote!(#t).to_string(),
        ),
    }
}
