use bevy::{
    asset::{AssetLoader, LoadContext, LoadedAsset},
    reflect::TypeUuid,
    utils::BoxedFuture,
};
use anyhow::{Error, anyhow};
use rogue_net::RogueNet;
use super::RogueNetAgent;

#[derive(TypeUuid)]
#[uuid = "5311f2d3-2c70-4490-bbfe-2f18cc885672"]
pub struct RogueNetAsset {
    pub agent: RogueNetAgent,
}

#[derive(Default)]
pub struct RogueNetAssetLoader;

impl AssetLoader for RogueNetAssetLoader {
    fn load<'a>(
        &'a self,
        bytes: &'a [u8],
        load_context: &'a mut LoadContext,
    ) -> BoxedFuture<'a, Result<(), Error>> {
        Box::pin(async move {
            let cursor = std::io::Cursor::new(bytes);
            let rogue_net = RogueNet::load_archive(cursor).map_err(|e| anyhow!(format!("{}", e)))?;
            let asset = RogueNetAsset { agent: RogueNetAgent { net: rogue_net } };
            load_context.set_default_asset(LoadedAsset::new(asset));
            Ok(())
        })
    }

    fn extensions(&self) -> &[&str] {
        &["roguenet"]
    }
}