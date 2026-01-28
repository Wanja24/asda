# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "marimo>=0.19.6",
#     "matplotlib==3.10.8",
#     "numpy==2.4.1",
#     "pandas==3.0.0",
#     "scikit-learn==1.8.0",
#     "seaborn==0.13.2",
#     "statsmodels==0.14.6",
# ]
# ///
import marimo

__generated_with = "0.18.4"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Regression builder

    Here you can do your own regression analysis on any dataset you have. Follow these steps.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Linear regression steps:**
    1. Import the data
    2. Clean the data
    3. Select variables
    4. Split the data
    5. Check assumptions: normal distribution of dependent variable? --> if not transform
    6. Check assumptions: multicollinearity (= redundancies among predictors)?
    --> throw out redundant variables
    7. Preprocess the data: scale the predictors
    8. Train linear regression on training set
    9. Predict on test set and evaluate with metrics (e.g. MAE, RMSE, MAPE, R2)
    10. Diagnostic plots
    - scatterplot with regression line
    - actual vs. predicted values
    - histogram of residuals --> normal distribution? if not, investigate why not?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Import data
    """)
    return


@app.cell
def _(mo):
    file_upload = mo.ui.file(
        filetypes=[".csv", ".txt", ".xls", ".xlsx", ".data"],
        label="Upload your data file"
    )
    file_upload
    return (file_upload,)


@app.cell
def _(file_upload):
    file_upload.value
    return


@app.cell
def _(mo):
    # --- Separator selection ---
    sep_input = mo.ui.text(
        label="Specify separator for CSV/TXT files (default is comma ,)",
        value=","
    )
    sep_input  # show the input widget
    return (sep_input,)


@app.cell
def _(BytesIO, Path, StringIO, file_upload, pd, sep_input):
    # --- File upload ---
    df = None

    if file_upload.value is not None:
        filename = file_upload.name()
        suffix = Path(filename).suffix.lower()
        data = file_upload.contents()

        # CSV
        if suffix in [".csv", ".data"]:
            df = pd.read_csv(BytesIO(data), sep=sep_input.value, header=None,  # file has no header row
            names=[
                "sepal length",
                "sepal width",
                "petal length",
                "petal width",
                "class",
            ])

        # TXT
        elif suffix == ".txt":
            df = pd.read_csv(StringIO(data.decode("utf-8")), sep=sep_input.value)

        # Excel
        elif suffix in [".xls", ".xlsx"]:
            df = pd.read_excel(BytesIO(data))

    # Return dataframe for display
    df
    return (df,)


@app.cell
def _(mo):
    mo.md(r"""
    ## Clean the data
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Select variables
    """)
    return


@app.cell
def _(df):
    df_num = df.select_dtypes(include="number")
    numeric_columns = df_num.columns.tolist()
    return (numeric_columns,)


@app.cell
def _(mo, numeric_columns):
    y_selector = mo.ui.dropdown(
        options=numeric_columns,
        label="Dependent variable (y)",
    )
    y_selector
    return (y_selector,)


@app.cell
def _(mo, numeric_columns):
    X_selector = mo.ui.multiselect(
        options=numeric_columns,
        label="Predictor variables (X)",
    )
    X_selector
    return (X_selector,)


@app.cell
def _(X_selector, df, pd, y_selector):
    # Get selected column names
    y_col = y_selector.value
    X_cols = X_selector.value or []

    # Make sure y is not in X
    if y_col in X_cols:
        X_cols = [col for col in X_cols if col != y_col]

    # Extract actual data from df
    if df is not None and y_col and X_cols:
        y = df[y_col]
        X = df[X_cols]
    else:
        y = None
        X = pd.DataFrame()  # empty dataframe if nothing selected
    return X, X_cols, y, y_col


@app.cell
def _(X, X_cols, df, mo, y, y_col):
    if df is None:
        mo.callout("Please upload a data file first.", kind="warning")
    elif not y_col:
        mo.callout("Please select a dependent variable.", kind="info")
    elif not X_cols:
        mo.callout("Please select at least one predictor variable.", kind="info")
    else:
        mo.callout(
            f"Ready to run regression with y = **{y}** and X = **{', '.join(X)}**",
            kind="success",
        )
    return


@app.cell
def _():
    #X = df.drop(columns=['Weight', 'Log_Weight', 'Root_Weight', 'Species'])
    #y = df[['Root_Weight']]
    #print(X.columns)
    #print(y.columns)
    return


@app.cell
def _(X, train_test_split, y):
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_test, X_train, y_test, y_train


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Check distributions
    """)
    return


@app.cell
def _(X_train, math, plt, sns):
    # initiating grid for histograms
    h_cols = 3 # number of columns
    h_rows = math.ceil(len(X_train.columns) / h_cols) # number of rows

    # create a grid of subplots
    fig, axes = plt.subplots(h_rows, h_cols, figsize = (10, 3 * h_rows))

    # flatten axes arry for easy iteration
    axes = axes.flatten()

    # iteratoe through numeric columns and create histogram
    for i, col in enumerate(X_train.columns):
        sns.histplot(data = X_train, x = col, kde = False, bins = 20, ax = axes[i])

        axes[i].set_xlabel(col)
        axes[i].set_ylabel('Frequency')

    # Turn off empty subplots if the number of plots is less than n_rows * n_cols
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    fig
    return


@app.cell
def _(mo):
    transform = mo.ui.dropdown(options=["none", "log", "square root"], value="none", label="Choose a transformation"
    )
    transform
    return (transform,)


@app.cell
def _(np, plt, sns, transform, y_test, y_train):
    # Transform y
    if transform.value == "log":
        y_train_transformed = np.log(y_train + 1)
        y_test_transformed = np.log(y_test + 1)
    elif transform.value == "square root":
        y_train_transformed = np.sqrt(y_train)
        y_test_transformed = np.sqrt(y_test)
    else:
        y_train_transformed = y_train
        y_test_transformed = y_test

    # Plot
    plt.figure(figsize=(6,4))
    distr = sns.histplot(data=y_train_transformed, kde=False, bins=20)

    # Fix x-axis label
    distr.set(
        xlabel=f"Weight ({transform.value})",
        ylabel="Count",
        title="Distribution of Transformed y (Training Set)"
    )
    distr
    return y_test_transformed, y_train_transformed


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Check redundancies
    """)
    return


@app.cell
def _(X_train, pd, variance_inflation_factor):
    # Assume X is the DataFrame of explanatory variables
    _vif_data = pd.DataFrame()
    _vif_data['Variable'] = X_train.columns
    _vif_data['VIF'] = [variance_inflation_factor(X_train.values, i).round(2) for i in range(X_train.shape[1])]
    _vif_data
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Scale predictors
    """)
    return


@app.cell
def _(StandardScaler, X_test, X_train, mo, pd):
    # Initialize scaler
    scaler = StandardScaler()

    # Fit on training data
    X_train_scaled = scaler.fit_transform(X_train)

    # Transform test data
    X_test_scaled = scaler.transform(X_test)

    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)

    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)

    # --- User message ---
    mo.callout("Predictors have been scaled using StandardScaler.", kind="success")
    return X_test_scaled, X_train_scaled


@app.cell
def _():
    # Regression including Species as dummy variables

    #X = df.drop(columns=['Weight', 'Log_Weight', 'Root_Weight'])
    #X = pd.get_dummies(X, columns=['Species'], drop_first=True)
    #X
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Train linear regression
    """)
    return


@app.cell
def _(
    LinearRegression,
    X_test_scaled,
    X_train_scaled,
    mo,
    pd,
    y_train_transformed,
):
    # Create linear regression object
    regr = LinearRegression()

    # Train the model using the training sets
    regr.fit(X_train_scaled, y_train_transformed)

    # Make predictions using the testing set
    y_pred = regr.predict(X_test_scaled)

    # Display coefficients as a table
    coef_df = pd.DataFrame({
        "Predictor": X_train_scaled.columns,
        "Coefficient": regr.coef_.ravel().round(2)
    })

    # Marimo callout for user info
    mo.callout("Linear regression model trained successfully. Here are the coefficients:", kind="success")
    return coef_df, y_pred


@app.cell
def _(coef_df):
    # Display coefficients table
    coef_df
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Evaluate linear regression
    """)
    return


@app.cell
def _(
    mean_absolute_error,
    mean_absolute_percentage_error,
    r2_score,
    root_mean_squared_error,
    y_pred,
    y_test_transformed,
):
    # Mean Absolute Error
    mae = mean_absolute_error(y_true = y_test_transformed, y_pred = y_pred) 

    # Mean Squared Error
    rmse = root_mean_squared_error(y_true = y_test_transformed,
                                   y_pred = y_pred) 

    # Mean Absolute Error %
    dumb_mape = mean_absolute_percentage_error(y_true = y_test_transformed,
                                               y_pred = y_pred) 

    # Rsquared
    r2 = r2_score(y_true = y_test_transformed, y_pred = y_pred)
    return dumb_mape, mae, r2, rmse


@app.cell
def _(X_cols, dumb_mape, mae, pd, r2, rmse, y_col):
    # Create a DataFrame for the table
    metrics_table = pd.DataFrame({
        "Metric": ["Mean Absolute Error (MAE)", 
                   "Root Mean Squared Error (RMSE)", 
                   "Mean Absolute Percentage Error (MAPE)", 
                   "R-squared (R²)"],
        f"{y_col} ~ {X_cols}": [mae, rmse, dumb_mape, r2],
    })

    # Display the table
    metrics_table
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Plots
    """)
    return


@app.cell
def _(X_test_scaled, math, plt, sns, y_col, y_pred, y_test_transformed):
    n_cols = 3
    n_rows = math.ceil(len(X_test_scaled.columns) / n_cols)

    fig2, _axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
    _axes = _axes.flatten()

    y_test_series = y_test_transformed.squeeze()
    y_pred_series = y_pred.ravel()

    for _i, predictor in enumerate(X_test_scaled.columns):
        ax = _axes[_i]
        sns.scatterplot(x=X_test_scaled[predictor], y=y_test_series, color="black", label="Actual Data", ax=ax)
        sns.lineplot(x=X_test_scaled[predictor], y=y_pred_series, color="blue", label="Regression Line", ax=ax)
        ax.set_xlabel(f"{predictor} (Scaled)")
        ax.set_ylabel(f"{y_col} (Transformed)")
        ax.set_title(f"{predictor} vs. {y_col}")
        ax.legend()

    # Turn off unused subplots
    for _j in range(_i+1, len(_axes)):
        _axes[_j].set_visible(False)

    plt.tight_layout()
    fig2  # Return a single figure — Marimo renders this cleanly

    return y_pred_series, y_test_series


@app.cell
def _(plt, sns, y_pred_series, y_test_series):
    # Create figure and axes
    fig3, ax3 = plt.subplots(figsize=(8, 6))

    # Scatter plot of actual vs predicted
    sns.scatterplot(
        x=y_test_series,
        y=y_pred_series,
        color="blue",
        label="Predicted vs Actual",
        ax=ax3
    )

    # Diagonal line for ideal fit
    min_val = min(y_test_series.min(), y_pred_series.min())
    max_val = max(y_test_series.max(), y_pred_series.max())
    ax3.plot([min_val, max_val], [min_val, max_val], color="red", linestyle="--", label="Ideal Fit")

    # Labels and title
    ax3.set_xlabel("Actual Values")
    ax3.set_ylabel("Predicted Values")
    ax3.set_title("Actual vs Predicted Values")
    ax3.legend()

    # Marimo-friendly display
    fig3

    return


@app.cell
def _(plt, sns, y_pred, y_test_transformed):
    # Calculate residuals
    residuals_series = y_test_transformed.squeeze() - y_pred.ravel()  # transformed y if used

    # Create figure and axes (new variable names)
    fig_resid, ax_resid = plt.subplots(figsize=(8, 6))

    # Plot histogram of residuals
    sns.histplot(residuals_series, bins=20, ax=ax_resid)

    # Labels and title
    ax_resid.set_xlabel("Residuals")
    ax_resid.set_ylabel("Frequency")
    ax_resid.set_title("Histogram of Residuals")

    # Marimo-friendly display
    fig_resid
    return


@app.cell
def _():
    import marimo as mo
    from io import BytesIO, StringIO
    from pathlib import Path
    import pandas as pd
    import numpy as np
    import math
    import seaborn as sns
    import matplotlib.pyplot as plt
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_absolute_error
    from sklearn.metrics import root_mean_squared_error
    from sklearn.metrics import mean_absolute_percentage_error
    from sklearn.metrics import r2_score
    from sklearn.linear_model import LinearRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.preprocessing import OneHotEncoder
    import statsmodels.formula.api as smf
    return (
        BytesIO,
        LinearRegression,
        Path,
        StandardScaler,
        StringIO,
        math,
        mean_absolute_error,
        mean_absolute_percentage_error,
        mo,
        np,
        pd,
        plt,
        r2_score,
        root_mean_squared_error,
        sns,
        train_test_split,
        variance_inflation_factor,
    )


if __name__ == "__main__":
    app.run()
