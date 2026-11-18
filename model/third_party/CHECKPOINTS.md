# Official checkpoint manifest

Downloaded binary files are stored outside the repository under
`~/zoujunhong/model/pretrain/`. This download provenance includes retained
historical pretraining weights; it does not register additional model families.

| Encoder | Official URL | Local path | Expected bytes | SHA-256 |
|---|---|---|---:|---|
| Utonia | `https://huggingface.co/Pointcept/Utonia/resolve/main/utonia.pth` | `~/zoujunhong/model/pretrain/utonia/utonia.pth` | 549168522 | `a0a2e5e234ce943082ae9ba26bcff74e285f16ed8d04fed5c36ffe8dc30ea353` |
| Uni3D-L | `https://huggingface.co/BAAI/Uni3D/resolve/main/modelzoo/uni3d-l/model.pt` | `~/zoujunhong/model/pretrain/uni3d_l/modelzoo/uni3d-l/model.pt` | 614861649 | `540a3cafac52c251cbda1844d109bc598f142b4f5a2118db87cd722b01800c20` |

The hashes are the upstream Hugging Face linked-object ETags and should match a
local `sha256sum` after a complete download.

DSAPT uses `~/zoujunhong/model/pretrain/utonia/utonia_tiny.pth`; the Tiny
architecture is recorded in `model/DSAPT/architecture.json`.
