# NFT Recommender system

This is the Pytorch implementation of the paper **NFTs to MARS: Multi-Attention Recommender System for NFTs**. 
All experiments were repeated three times, which can be replicated with three different random seeds (2022, 2023, 2024).<br>
<br>
Explore our [Data Description](Data_description.md) for detailed data information, and [Experimental Details](Experimental_details.md) for the detailed experiment settings.








## Get started

### **`Our model`**

1. Install Python 3.10.9. The required packages are as follows:

```
numpy==1.25.0
scikit-learn==1.2.2
scipy==1.10.1
torch==2.0.0
torch-geometric==1.7.2
torch-cluster==1.6.1
torch-scatter==2.1.1
torch-sparse==0.6.17
torch-spline-conv==1.2.2
torchcontrib==0.0.2
```

2. Download data. You can obtain all pre-processed data from [Google Drive](https://drive.google.com/drive/folders/12WeTJ6HzjGI0giirlu__PFSGtxno7cWU?usp=share_link).
   
   (For detailed description about the data, please refer to `Our model/Create_dataset.ipynb`)

3. Create a directory `dataset/collections` and place the downladed data in that location. 

4. Download requirement packages ***pip install -r requirements.txt***

5. Train the model. 

   We provide the experiment scripts of all datasets in the file `run.sh`. You can reproduce the experiment results by: 

   ~~~bash
   bash run.sh
   ~~~

6. *(Ablation studies)* Train the model using a single graph. 

   In this case, since the multi-modal attention used in the existing NFT-MARS model cannot be applied, the model name has been changed to "MO", which stands for Multi Objective. For example, "MO_v" is a single graph model that utilizes visual features. 

   We provide the experiment scripts of all datasets in the file `run_MO.sh`. You can reproduce the experiment results by:

   ```bash
   bash run_MO.sh
   ```

## Contribution

- We develop a model to address three unique challenges that NFT recommender systems face. Our method consists of three key components:
  1. **Graph attention** to handle extremely sparse user-item interactions
  2. **Multi-modal attention** to incorporate user-specific feature preferences
  3. **Multi-task learning** to address the dual nature of NFTs as artworks and investment assets
- We demonstrate the effectiveness of our model compared to various baseline models using the actual transaction data of NFTs collected directly from blockchain for four of the most popular NFT collections.
- We constructed a dataset by combining this transaction data with hand-crafted features, which can be used as a benchmark dataset for any NFT recommendation model. Datasets are available on [Google Drive](https://drive.google.com/drive/folders/12WeTJ6HzjGI0giirlu__PFSGtxno7cWU?usp=share_link).

## Baselines

Our repository includes two additional folders: "**Baseline models (MGAT)**" and "**Baseline models (Others)**". All baselines except for MGAT were implemented using RecBole, so they are separated into their own folder.

### **`Baseline models (MGAT)`**

contains the code to implement the MGAT model.

You can follow the steps in the "Get started" section above, but note that the experiment script names are different. 

```bash
bash run_MGAT.sh
```

### **`Baseline models (Others)`**

contains code to implement other models including Pop, ItemKNN, BPR, DMF, LightGCN, FM, DeepFM, WideDeep, DCN, and AutoInt. 

You can follow the steps in the "Get started" section above, but note that you need to install different version of Python. 

```
Python 3.7.12
```

## Acknowledgement

We appreciate the following github repos a lot for their valuable code base or datasets:

1. [RecBole](https://github.com/RUCAIBox/RecBole)
2. [MGAT](https://github.com/zltao/MGAT)
