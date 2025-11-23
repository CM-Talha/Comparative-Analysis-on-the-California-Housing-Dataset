Environment Requirements:
1. numpy
2. pandas
3. matplotlib
4. scikit-learn

Dataset Information:
1. California Housing dataset

Note: No separate dataset files are required to run the notebook.

Instructions to Run the Notebook:
1. Open the Notebook:Run the provided code file on google colab named MF_AI_Project_California.ipynb.
2. Execute Cells Sequentially: Run all cells in the notebook in order. 
	This will perform the following steps:
		a. Load the data.
		b. Preprocess the data (scaling, train/test split, adding an intercept column).
		c. Compute linear regression solutions using Ordinary Least Squares and SVD.
		d. Implement and run Batch Gradient Descent and Adam Optimizer.
		e. Perform Principal Component Analysis (PCA) for dimensionality reduction.
		f. Generate various plots (Actual vs Predicted, Residuals vs Predicted, Singular Values, Loss vs Iterations, Scree Plot, Test MSE vs k).
		g. Display a final comparison table of MSE and R2 scores for all methods.
