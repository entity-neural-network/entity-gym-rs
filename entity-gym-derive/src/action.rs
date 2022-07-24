use proc_macro2::TokenStream;
use proc_macro_error::abort_call_site;
use quote::quote;
use syn::{parse_macro_input, Data, DeriveInput, Ident};

pub fn derive(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    let input = parse_macro_input!(input as DeriveInput);

    let name = &input.ident;
    let (impl_generics, ty_generics, where_clause) = input.generics.split_for_impl();

    let (from_u64, to_u64, num_actions, labels) = generate(name, &input.data);
    let name_str = name.to_string();

    let expanded = quote! {
        #[allow(unused_qualifications)]
        impl #impl_generics Action<'static> for #name #ty_generics #where_clause {
            fn from_u64(index: u64) -> Self{
                #from_u64
            }

            fn to_u64(&self) -> u64 {
                #to_u64
            }

            fn num_actions() -> u64 {
                #num_actions
            }

            fn name() -> &'static str {
                #name_str
            }

            fn labels() -> &'static [&'static str] {
                #labels
            }
        }
    };

    // Hand the output tokens back to the compiler
    proc_macro::TokenStream::from(expanded)
}

fn generate(ident: &Ident, data: &Data) -> (TokenStream, TokenStream, TokenStream, TokenStream) {
    match data {
        Data::Enum(em) => {
            let len = em.variants.len() as u64;
            let mut from_u64 = vec![];
            let mut to_u64 = vec![];
            let mut labels = vec![];
            for (i, variant) in em.variants.iter().enumerate() {
                let i = i as u64;
                let variant_ident = &variant.ident;
                let variant_name = &variant.ident.to_string();
                from_u64.push(quote! {
                    #i => { #ident::#variant_ident }
                });
                to_u64.push(quote! {
                    #ident::#variant_ident => { #i }
                });
                labels.push(quote! {
                    #variant_name
                });
            }
            (
                quote! {
                    match index {
                        #(#from_u64),*
                        _ => panic!("Invalid action index: {}", index),
                    }
                },
                quote! {
                    match &self {
                        #(#to_u64),*
                    }
                },
                quote! { #len },
                quote! { &[#(#labels),*] },
            )
        }
        _ => abort_call_site!("\"Action\" can only be derived for enums"),
    }
}
