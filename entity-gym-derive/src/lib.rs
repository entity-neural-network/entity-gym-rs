use proc_macro_error::proc_macro_error;

mod action;
mod featurizable;

#[proc_macro_error]
#[proc_macro_derive(Featurizable)]
pub fn derive_featurizable(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    featurizable::derive(input)
}

#[proc_macro_error]
#[proc_macro_derive(Action)]
pub fn derive_action(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    action::derive(input)
}
