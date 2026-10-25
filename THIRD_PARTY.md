# Public sources, licenses and adaptation boundary

- HINT-SD: https://github.com/wgcyeo/HINT-SD ,
  commit bc95b44192b06328a88773dbf3b345fa1c499093, Apache-2.0.
  Paper: https://arxiv.org/abs/2605.17873 .
  Original selection, reverse-KL/top100+tail, backward, optimizer, schedule,
  LoRA setup and EMA stay unmodified. Only environment/feedback domain,
  synchronous inference transport, fixed140 evaluation and local telemetry are
  adapted. This is not author-reported WebShop results or a GRPO/RETHINK hybrid.
  The public uv.lock is fetched unmodified.
- SDAR / verl-agent: https://github.com/ZJU-REAL/SDAR ,
  commit29dcb12a55d8fc17ba6ae03fd0a69048f43051f3, Apache-2.0.
  Native WebShop engine, prompt/history, parser, reward and search only.
  Exact native worker/formatter/memory ASTs execute without importing the Ray
  training stack. No upstream tracked source is patched. Content-ID filtering
  excludes the common140 objectives before Actor interaction.
- WebShop: https://github.com/princeton-nlp/WebShop , MIT.
  Vendored license remains at
  .deps/sdar/agent_system/environments/env_package/webshop/webshop/LICENSE.md .
  Product assets retain source terms, are downloaded, not redistributed or
  relicensed by this wrapper.
- Public assets: https://huggingface.co/datasets/HongbangYuan/webshop and
  https://huggingface.co/datasets/YWZBrandon/webshop-data .
  Revisions/SHA in configs/assets.json; mirror bytes match. This is not an
  independently recovered author historical Google Drive checksum.
- spaCy en_core_web_sm3.7.1:
  https://github.com/explosion/spacy-models/releases/tag/en_core_web_sm-3.7.1 ,
  MIT. Wheel URL/SHA pinned.
- Qwen/Qwen2.5-1.5B-Instruct:
  https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct , Apache-2.0.
  Qwen/Qwen2.5-3B-Instruct:
  https://huggingface.co/Qwen/Qwen2.5-3B-Instruct , Qwen Research License.
  Weights downloaded from pinned revisions, not shipped in Git.

No maintainer local file, private dataset/server, credential, API key or W&B
account is required. Credit HINT-SD, SDAR/verl-agent, WebShop, Qwen and dependency
authors. This independent adaptation does not imply endorsement.
