# Data Analysis and Visualization Assignment
# Using the Iris dataset from sklearn

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris

# Set seaborn style for better looking plots
sns.set(style="whitegrid")

def load_data():
    """Load the Iris dataset and handle any potential errors"""
    try:
        iris = load_iris()
        data = pd.DataFrame(iris.data, columns=iris.feature_names)
        data['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)
        return data
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None

def explore_data(df):
    """Perform basic data exploration"""
    if df is None:
        return
    
    print("=== Dataset Overview ===")
    print(f"Shape: {df.shape}")
    print("\nFirst 5 rows:")
    print(df.head())
    
    print("\n=== Basic Statistics ===")
    print(df.describe())
    
    print("\n=== Species Distribution ===")
    print(df['species'].value_counts())
    
    print("\n=== Missing Values ===")
    print(df.isnull().sum())

def create_visualizations(df):
    """Create various visualizations of the data"""
    if df is None:
        return
    
    # Set up the figure layout
    plt.figure(figsize=(15, 10))
    
    # 1. Box plots for each feature by species
    plt.subplot(2, 2, 1)
    sns.boxplot(x='species', y='sepal length (cm)', data=df)
    plt.title('Sepal Length by Species')
    
    plt.subplot(2, 2, 2)
    sns.boxplot(x='species', y='sepal width (cm)', data=df)
    plt.title('Sepal Width by Species')
    
    plt.subplot(2, 2, 3)
    sns.boxplot(x='species', y='petal length (cm)', data=df)
    plt.title('Petal Length by Species')
    
    plt.subplot(2, 2, 4)
    sns.boxplot(x='species', y='petal width (cm)', data=df)
    plt.title('Petal Width by Species')
    
    plt.tight_layout()
    plt.show()
    
    # 2. Pairplot to visualize relationships between features
    print("\nCreating pairplot (this may take a moment)...")
    sns.pairplot(df, hue='species', palette='viridis')
    plt.suptitle('Pairplot of Iris Features by Species', y=1.02)
    plt.show()
    
    # 3. Correlation heatmap
    plt.figure(figsize=(8, 6))
    numeric_df = df.select_dtypes(include=['float64'])
    sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', center=0)
    plt.title('Feature Correlation Heatmap')
    plt.show()

def main():
    """Main function to execute the analysis"""
    print("Starting Iris dataset analysis...\n")
    
    # Load the data
    iris_data = load_data()
    
    # Explore the data
    explore_data(iris_data)
    
    # Create visualizations
    create_visualizations(iris_data)
    
    print("\nAnalysis complete!")

if __name__ == "__main__":
    main()