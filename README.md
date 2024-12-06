<p align="center">
  <h1 align="center">SeeGround: See and Ground for Zero-Shot Open-Vocabulary 3D Visual Grounding</h1>

<p align="center">
<a href="https://rongli.tech/">Rong Li</a>,
<a href="https://sj-li.com/">Shijie Li</a>,
<a href="https://ldkong.com/">Lingdong Kong</a>,
<a href="https://dawdleryang.github.io/">Xulei Yang</a>,
<a href="https://junweiliang.me/">Junwei Liang</a>
</p>


<p align="center">
    <a href='https://arxiv.org/abs/2412.04383'>
      <img src='https://img.shields.io/badge/Paper-PDF-red?style=flat&logo=arXiv&logoColor=red' alt='Paper PDF'>
    </a>
    <a href='https://seeground.github.io/' style='padding-left: 0.5rem;'>
      <img src='https://img.shields.io/badge/Project-Page-blue?style=flat&logo=Google%20chrome&logoColor=blue' alt='Project Page'>
    </a>
  </p>

</p>



# 
<p align="center">
  <img src="./figs/teaser.jpg" alt="Logo" width="100%">
</p>

3D Visual Grounding (3DVG) aims to locate objects in 3D scenes based on textual descriptions, which is essential for applications like augmented reality and robotics. Traditional 3DVG approaches rely on annotated 3D datasets and predefined object categories, limiting scalability and adaptability. To overcome these limitations, we introduce SeeGround, a zero-shot 3DVG framework leveraging 2D Vision-Language Models (VLMs) trained on large-scale 2D data. We propose to represent 3D scenes as a hybrid of query-aligned rendered images and spatially enriched text descriptions, bridging the gap between 3D data and 2D-VLMs input formats. We propose two modules: the Perspective Adaptation Module, which dynamically selects viewpoints for query-relevant image rendering, and the Fusion Alignment Module, which integrates 2D images with 3D spatial descriptions to enhance object localization. Extensive experiments on ScanRefer and Nr3D demonstrate that our approach outperforms existing zero-shot methods by large margins. Notably, we exceed weakly supervised methods and rival some fully supervised ones, outperforming previous SOTA by 7.7% on ScanRefer and 7.1% on Nr3D, showcasing its effectiveness.


# Overview 


<p align="center">
  <img src="./figs/arch_1.jpg" alt="Logo" width="100%">
</p>

Overview of the <span style="color: #FF8E26;"><b>See</b></span><span
  style="color: #01B1A0;"><b>Ground</b></span> framework. We first use a 2D-VLM to interpret the query,
identifying both the target object (e.g., "laptop") and a context-providing anchor (e.g., "chair with floral
pattern"). A dynamic viewpoint is then selected based on the anchor’s position, enabling the capture of a 2D
rendered image that aligns with the query’s spatial requirements. Using the Object Lookup Table (OLT), we
retrieve the 3D bounding boxes of relevant objects, project them onto the 2D image, and apply visual prompts
to mark visible objects, filtering out occlusions. The image with prompts, along with the spatial
descriptions and query, are then input into the 2D-VLM for precise localization of the target object.
Finally, the 2D-VLM outputs the target object’s ID, and we retrieve its 3D bounding box from the OLT to
provide the final, accurate 3D position in the scene.


# 1. Environment Setup

We recommend using the [official Docker image](https://hub.docker.com/r/qwenllm/qwenvl) for environment setup
```
docker pull qwenllm/qwenvl
```

# 2. Download Model Weights


You can download the qwen2-vl model weights from either of the following sources:
- [huggingface](https://huggingface.co/collections/Qwen/qwen2-vl-66cee7455501d7126940800d) 
- [modelscope](https://modelscope.cn/collections/Qwen2-VL-b4ce472a80274b)

# 3. Download Datasets

## 3.1. ScanRefer 

Download ScanRefer dataset from [official repo](https://github.com/daveredrum/ScanRefer), and place it in the following directory:
```
data/ScanRefer/ScanRefer_filtered_val.json
```

## 3.2. Nr3D 

Download the Nr3D dataset from the [official repo](https://github.com/referit3d/referit3d), and place it in the following directory:

```
data/Nr3D/Nr3D.jsonl
```

## 3.3. Vil3dref Preprocessed Data

Download the preprocessed Vil3dref data from [vil3dref](https://github.com/cshizhe/vil3dref).

The expected structure should look like this:
```
referit3d/
.
├── annotations
|   ├── meta_data
|   │   ├── cat2glove42b.json
|   │   ├── scannetv2-labels.combined.tsv
|   │   └── scannetv2_raw_categories.json
│   └── ...
├── ...
└── scan_data
    ├── ...
    ├── instance_id_to_name
    └── pcd_with_global_alignment
```

# 4.Data Processing

Download [mask3d pred](https://github.com/CurryYuan/ZSVG3D) first.

- ScanRefer 
```
python prepare_data/object_lookup_table_scanrefer.py
```

- Nr3D

```
python prepare_data/process_feat_3d.py

python prepare_data/object_lookup_table_nr3d.py
```

Alternatively, you can download the preprocessed Object Lookup Table from TODO.


# 5. Inference

## 5.1. Deploying VLM Service

We use `vllm` for deploying vlm. It is recommended to run the following command in a `tmux` session on your server:

```
python -m vllm.entrypoints.openai.api_server --model /your/qwen2-vl-model/path  --served-model-name Qwen2-VL-72B-Instruct --tensor_parallel_size=8
```

The `--tensor_parallel_size` flag controls the number of GPUs required. Adjust it according to your memory resources.

## 5.2. Generating Anchors/Targets

- ScanRefer
```
python parse_query/generate_query_data_scanrefer.py
```

- Nr3D
```
python parse_query/generate_query_data_nr3d.py
```

## 5.3. Prediction

- ScanRefer
```
python inference/inference_scanrefer.py
```

- Nr3D
```
python inference/inference_nr3d.py
```

## 5.4. Evaluation

- ScanRefer 
```
python eval/eval_nr3d.py
```

- Nr3D
```
python eval/eval_scanrefer.py
```

# License
This work is released under the Apache 2.0 license.

# Citation 

If you find this work and code repository helpful, please consider starring it and citing the following paper:

```
@inproceedings{li2024seeground,
  title     = {SeeGround: See and Ground for Zero-Shot Open-Vocabulary 3D Visual Grounding},
  author    = {Rong Li and Shijie Li and Lingdong Kong and Xulei Yang and Junwei Liang},
  journal   = {arXiv},
  year      = {2024},
}
```

# Acknowledgements

We would like to thank the following repositories for their contributions:
- [ZSVG3D](https://github.com/CurryYuan/ZSVG3D)
- [ReferIt3D](https://github.com/referit3d/referit3d)
- [Vil3dref](https://github.com/cshizhe/vil3dref)
- [OpenIns3D](https://github.com/Pointcept/OpenIns3D)