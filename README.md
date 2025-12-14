# Presentation Attack Detection (PAD) for Identity Documents

## 🎯 Project Goal

The goal of this project is to study and develop methods for **Presentation Attack Detection (PAD)** in the context of **identity documents** (ID cards, passports, etc.).

Presentation Attack Detection aims to distinguish between **genuine document presentations** and **fraudulent attempts**, such as:
- Printed document copies
- Screen recaptures (smartphone, tablet, laptop)
- Replays or other spoofing scenarios

The project focuses on building a reliable pipeline that supports PAD research by enabling robust document analysis under different acquisition conditions. This work serves as a foundation for developing, evaluating, and improving PAD models for document security applications.

---

## 📊 Dataset Source

The dataset used in this project is **DLC-2021**, a public dataset designed for document analysis and **Presentation Attack Detection research**.

🔗 **Official dataset link**:  
https://zenodo.org/records/7467028

### Dataset overview:
- 1,424 videos of identity documents  
- Multiple PAD-related scenarios (recaptures, copies, grayscale, etc.)  
- Diverse acquisition conditions and document types  
- Fully synthetic and GDPR-compliant  

---

## 🧪 Dataset Sampling in This Repository

⚠️ **Important note**

The full DLC-2021 dataset is large and cannot be fully included in this GitHub repository.

For this reason:
- Only a **small sample of 3 images** is included
- The **original dataset organization is preserved**
- File formats and naming conventions remain unchanged

This keeps the repository lightweight while allowing seamless extension to the complete dataset.

---

## 🚀 Usage

This repository can be used to:
- Explore and experiment with PAD approaches for identity documents
- Serve as a base for PAD model development and evaluation
- Support research on document spoofing and fraud detection
- Extend experiments to the full DLC-2021 dataset

To work with the complete dataset, simply download it from Zenodo and replace the sample data while keeping the same organization.

---

## 📝 Credits & License

- Dataset: **DLC-2021 (Zenodo)**
- Intended for **academic and research purposes**
