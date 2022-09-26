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

            fn labels() -> Vec<String> {
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
                    #variant_name.to_string()
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
                quote! { vec![#(#labels),*] },
            )
        }
        Data::Struct(data) => match &data.fields {
            syn::Fields::Named(fields) => {
                let mut sub_indices = vec![];
                let mut decode = vec![];
                let mut encode = vec![];
                let mut num_actions = vec![];
                let mut name_type = vec![];
                for field in fields.named.iter() {
                    let name = field.ident.as_ref();
                    let name_str = field.ident.as_ref().unwrap().to_string();
                    let typ = &field.ty;
                    sub_indices.push(quote! {
                        let #name = index % #typ::num_actions();
                        index /= #typ::num_actions();
                    });
                    decode.push(quote! {
                        #name: #typ::from_u64(#name)
                    });

                    encode.push(quote! {
                        encoded = encoded * #typ::num_actions() + self.#name.to_u64();
                    });

                    num_actions.push(quote! {
                        num_actions *= #typ::num_actions()
                    });

                    name_type.push((typ, name_str));
                }
                sub_indices.reverse();

                let mut labels = quote! { _result.push(labels.join(",")); };
                for (typ, name_str) in name_type.iter().rev() {
                    labels = quote! {
                        for s in #typ::labels() {
                            labels.push(format!("{}={}", #name_str, s));
                            #labels
                            labels.pop();
                        }
                    };
                }

                (
                    quote! {
                        let mut index = index;
                        #(#sub_indices)*
                        #ident { #(#decode,)* }
                    },
                    quote! {
                        let mut encoded = 0;
                        #(#encode;)*
                        encoded
                    },
                    quote! {
                        let mut num_actions = 1;
                        #(#num_actions;)*
                        num_actions
                    },
                    quote! {
                        let mut _result = vec![];
                        let mut labels = vec![];
                        #labels;
                        _result
                    },
                )
            }
            syn::Fields::Unnamed(_fields) => todo!(),
            syn::Fields::Unit => todo!(),
        },
        _ => abort_call_site!("\"Action\" can only be derived for enums and structs"),
    }
}
