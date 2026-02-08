"""
Rule-Based and Tree-Based Models Module
Implements: Association Rules, Decision Trees, Random Forest, ID3, CHAID
(From Unit 3.1 and 3.2 of syllabus)
"""

import numpy as np
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, plot_tree
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import matplotlib.pyplot as plt


class AssociationRuleMining:
    """
    Association Rule Mining: Apriori, Eclat, FP-Growth
    (From Unit 3.1 of syllabus)
    
    Note: For production use, consider mlxtend library for full implementation
    This is a simplified demonstration
    """
    
    def __init__(self, min_support=0.5, min_confidence=0.7):
        self.min_support = min_support
        self.min_confidence = min_confidence
        self.frequent_itemsets = []
        self.rules = []
    
    def apriori_algorithm(self, transactions):
        """
        Simplified Apriori Algorithm for Association Rule Mining
        transactions: list of sets/lists
        """
        print("\nRunning Apriori Algorithm...")
        print(f"Min Support: {self.min_support}, Min Confidence: {self.min_confidence}")
        
        # Get all unique items
        all_items = set()
        for transaction in transactions:
            all_items.update(transaction)
        
        # Find frequent 1-itemsets
        item_counts = {}
        for item in all_items:
            count = sum(1 for transaction in transactions if item in transaction)
            support = count / len(transactions)
            if support >= self.min_support:
                item_counts[frozenset([item])] = support
        
        self.frequent_itemsets = item_counts
        print(f"Found {len(self.frequent_itemsets)} frequent itemsets")
        
        # Generate association rules
        self._generate_rules(transactions)
        
        return self.frequent_itemsets, self.rules
    
    def _generate_rules(self, transactions):
        """Generate association rules from frequent itemsets"""
        self.rules = []
        
        for itemset, support in self.frequent_itemsets.items():
            if len(itemset) < 2:
                continue
            
            for item in itemset:
                antecedent = itemset - frozenset([item])
                consequent = frozenset([item])
                
                # Calculate confidence
                antecedent_count = sum(1 for t in transactions if antecedent.issubset(t))
                if antecedent_count > 0:
                    confidence = sum(1 for t in transactions if itemset.issubset(t)) / antecedent_count
                    
                    if confidence >= self.min_confidence:
                        lift = confidence / support
                        self.rules.append({
                            'antecedent': antecedent,
                            'consequent': consequent,
                            'support': support,
                            'confidence': confidence,
                            'lift': lift
                        })
        
        print(f"Generated {len(self.rules)} association rules")
        
        # Display top rules
        if self.rules:
            print("\nTop Association Rules:")
            for i, rule in enumerate(sorted(self.rules, key=lambda x: x['confidence'], reverse=True)[:5]):
                print(f"{i+1}. {set(rule['antecedent'])} => {set(rule['consequent'])}")
                print(f"   Support: {rule['support']:.3f}, Confidence: {rule['confidence']:.3f}, Lift: {rule['lift']:.3f}")


class DecisionTreeModel:
    """
    Decision Trees for Classification and Regression
    (From Unit 3.2: Decision Trees, ID3, CHAID)
    """
    
    def __init__(self, task='classification', criterion=None, max_depth=None, 
                 min_samples_split=2, min_samples_leaf=1):
        """
        task: 'classification' or 'regression'
        criterion: 'gini', 'entropy' for classification; 'squared_error' for regression
        """
        self.task = task
        self.max_depth = max_depth
        
        if task == 'classification':
            self.criterion = criterion if criterion else 'gini'
            self.model = DecisionTreeClassifier(
                criterion=self.criterion,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                random_state=42
            )
        else:  # regression
            self.criterion = criterion if criterion else 'squared_error'
            self.model = DecisionTreeRegressor(
                criterion=self.criterion,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                random_state=42
            )
        
        self.is_trained = False
        self.feature_names = None
        self.class_names = None
    
    def train(self, X_train, y_train, feature_names=None, class_names=None):
        """Train decision tree"""
        print(f"\nTraining Decision Tree ({self.task})...")
        print(f"Criterion: {self.criterion}, Max Depth: {self.max_depth}")
        
        self.model.fit(X_train, y_train)
        self.is_trained = True
        self.feature_names = feature_names
        self.class_names = class_names
        
        print("Decision Tree training completed!")
        
        # Display tree information
        print(f"Tree depth: {self.model.get_depth()}")
        print(f"Number of leaves: {self.model.get_n_leaves()}")
        
        return self.model
    
    def predict(self, X):
        """Make predictions"""
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """Get probability estimates (classification only)"""
        if self.task != 'classification':
            raise ValueError("Probability prediction only available for classification")
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        return self.model.predict_proba(X)
    
    def get_feature_importance(self):
        """Get feature importance scores"""
        if not self.is_trained:
            raise ValueError("Model must be trained first")
        
        importances = self.model.feature_importances_
        
        if self.feature_names:
            feature_importance = dict(zip(self.feature_names, importances))
            sorted_features = sorted(feature_importance.items(), 
                                   key=lambda x: x[1], reverse=True)
            
            print("\nFeature Importances:")
            for feature, importance in sorted_features[:10]:
                print(f"{feature}: {importance:.4f}")
        
        return importances
    
    def visualize_tree(self, max_depth_display=3):
        """
        Visualize the decision tree
        (Demonstrates ID3-like tree structure)
        """
        if not self.is_trained:
            raise ValueError("Model must be trained first")
        
        plt.figure(figsize=(20, 10))
        plot_tree(self.model, 
                 feature_names=self.feature_names,
                 class_names=self.class_names,
                 filled=True,
                 rounded=True,
                 max_depth=max_depth_display,
                 fontsize=10)
        plt.title(f'Decision Tree Visualization (max depth shown: {max_depth_display})', 
                 fontsize=16)
        plt.tight_layout()
        plt.savefig('./decision_tree_visualization.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Decision tree visualization saved (showing depth up to {max_depth_display})")


class RandomForestModel:
    """
    Random Forest Classifier
    (From Unit 3.2: Random Forest Classifier)
    """
    
    def __init__(self, task='classification', n_estimators=100, max_depth=None,
                 min_samples_split=2, min_samples_leaf=1, max_features='sqrt'):
        """
        task: 'classification' or 'regression'
        n_estimators: number of trees in the forest
        """
        self.task = task
        self.n_estimators = n_estimators
        
        if task == 'classification':
            self.model = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                max_features=max_features,
                random_state=42,
                n_jobs=-1
            )
        else:  # regression
            self.model = RandomForestRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                max_features=max_features,
                random_state=42,
                n_jobs=-1
            )
        
        self.is_trained = False
        self.feature_names = None
    
    def train(self, X_train, y_train, feature_names=None):
        """Train random forest"""
        print(f"\nTraining Random Forest ({self.task})...")
        print(f"Number of trees: {self.n_estimators}")
        
        self.model.fit(X_train, y_train)
        self.is_trained = True
        self.feature_names = feature_names
        
        print("Random Forest training completed!")
        
        return self.model
    
    def predict(self, X):
        """Make predictions"""
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """Get probability estimates (classification only)"""
        if self.task != 'classification':
            raise ValueError("Probability prediction only available for classification")
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        return self.model.predict_proba(X)
    
    def get_feature_importance(self, plot=True):
        """Get and plot feature importance"""
        if not self.is_trained:
            raise ValueError("Model must be trained first")
        
        importances = self.model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        print("\nTop 10 Most Important Features:")
        for i in range(min(10, len(importances))):
            if self.feature_names:
                print(f"{i+1}. {self.feature_names[indices[i]]}: {importances[indices[i]]:.4f}")
            else:
                print(f"{i+1}. Feature {indices[i]}: {importances[indices[i]]:.4f}")
        
        if plot and len(importances) > 0:
            plt.figure(figsize=(10, 6))
            
            # Plot top 20 features
            n_features_to_plot = min(20, len(importances))
            plt.barh(range(n_features_to_plot), importances[indices[:n_features_to_plot]])
            
            if self.feature_names:
                plt.yticks(range(n_features_to_plot), 
                          [self.feature_names[i] for i in indices[:n_features_to_plot]])
            else:
                plt.yticks(range(n_features_to_plot), 
                          [f'Feature {i}' for i in indices[:n_features_to_plot]])
            
            plt.xlabel('Importance', fontsize=12)
            plt.ylabel('Features', fontsize=12)
            plt.title('Random Forest Feature Importance', fontsize=14)
            plt.gca().invert_yaxis()
            plt.tight_layout()
            plt.savefig('./random_forest_feature_importance.png', 
                       dpi=300, bbox_inches='tight')
            plt.close()
            print("Feature importance plot saved")
        
        return importances
    
    def get_tree_depths(self):
        """Get statistics about tree depths in the forest"""
        if not self.is_trained:
            raise ValueError("Model must be trained first")
        
        depths = [tree.get_depth() for tree in self.model.estimators_]
        
        print(f"\nTree Depth Statistics:")
        print(f"Mean depth: {np.mean(depths):.2f}")
        print(f"Max depth: {np.max(depths)}")
        print(f"Min depth: {np.min(depths)}")
        
        return depths
