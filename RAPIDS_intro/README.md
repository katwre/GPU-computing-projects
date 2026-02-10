# Introduction to RAPIDS¹

RAPIDS is an open-source GPU-accelerated data science and analytics platform developed mainly by NVIDIA. Think of it as a way to run the Python data science stack (pandas, scikit-learn, etc.) on GPUs instead of CPUs, often getting 10-100× speedups on large datasets.

Its core libraries include:

- **cuDF** for a GPU version of pandas for dataframe operations (check out `Cudf_vs_pandas.ipynb`), 
- **cuML**  for GPU implementations of common machine-learning algorithms like PCA, UMAP, clustering, and regression (check out `RandomForest_cuML_vs_scikitlearn.ipynb` and `Cuml_clustering_models.ipynb`), 
- **cuGraph** for GPU graph analytics, and 
- integrations with **Dask** for scaling across multiple GPUs or machines. 
    
The key idea is that we can keep almost the same Python workflow and syntax while massively accelerating data manipulation, classical machine learning, and graph analysis, making RAPIDS especially useful for large-scale analytics such as genomics, single-cell data, and other high-dimensional scientific datasets. 

**skrub** is an open-source Python library that bridges the gap between messy tabular data and machine learning models. It can be used to show that cuML works with standard scikit-learn ecosystem tools - we can use skrub for preprocessing and then swap in cuML's GPU-accelerated models for huge speedups (check out `RandomForest_cuML_vs_scikitlearn.ipynb`).








--------


¹Implemented as part of the tutorials at the [PyData Global conference 2025](https://pydata.org/global2025)