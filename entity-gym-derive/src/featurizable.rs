use proc_macro2::TokenStream;
use proc_macro_error::abort;
use quote::quote;
use syn::{parse_macro_input, Data, DeriveInput, Field, Type};

pub fn derive(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    // Parse the input tokens into a syntax tree
    let input = parse_macro_input!(input as DeriveInput);

    let name = &input.ident;
    let (impl_generics, ty_generics, where_clause) = input.generics.split_for_impl();

    let (num_feats, features_names, featurize) = field_names(&input.data);
    let name_str = name.to_string();

    // Build the output, possibly using quasi-quotation
    let expanded = quote! {
        #[allow(unused_qualifications)]
        impl #impl_generics Featurizable for #name #ty_generics #where_clause {
            fn num_feats() -> usize {
                #num_feats
            }

            fn feature_names() -> &'static [&'static str] {
                #features_names
            }

            fn featurize(&self) -> Vec<f32> {
                #featurize
            }

            fn name() -> &'static str {
                #name_str
            }
        }
    };

    // Hand the output tokens back to the compiler
    proc_macro::TokenStream::from(expanded)
}

fn field_names(data: &Data) -> (TokenStream, TokenStream, TokenStream) {
    match data {
        Data::Struct(data) => match &data.fields {
            syn::Fields::Named(fields) => {
                let mut names = vec![];
                let count = fields.named.len();
                let mut features = vec![];
                for field in fields.named.iter() {
                    names.push(field.ident.as_ref().unwrap().to_string());
                    features.push(field_to_f32(field))
                }

                (
                    quote! { #count },
                    quote! { &[#(#names),*] },
                    quote! { vec![#(#features),*] },
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
            syn::Fields::Unit => (quote! { 0 }, quote! { &[] }, quote! { ::std::Vec::new() }),
        },
        Data::Enum(_) => unimplemented!("Enum not supported, must be struct"),
        Data::Union(_) => unimplemented!("Union not supported, must be struct"),
    }
}

fn field_to_f32(field: &Field) -> TokenStream {
    let ident = field.ident.as_ref().unwrap();
    match &field.ty {
        Type::Path(ty) => {
            let ty = ty.path.segments.last().unwrap();
            match ty.ident.to_string().as_str() {
                "f32" => quote! { self.#ident },
                "f64" | "u8" | "u16" | "u32" | "u64" | "i8" | "i16" | "i32" | "i64" => {
                    quote! { self.#ident as f32 }
                }
                "bool" => quote! { if self.#ident { 1.0 } else { 0.0 } },
                _ => abort!(
                    ty,
                    "Can't derive Featurizable for type `{}`",
                    quote!(#ty).to_string(),
                ),
            }
        }
        t => abort!(
            t,
            "Can't derive Featurizable for type `{}`",
            quote!(#t).to_string(),
        ),
    }
}
