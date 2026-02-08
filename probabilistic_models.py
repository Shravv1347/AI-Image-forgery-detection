"""
Probabilistic Models Module
Implements: Naive Bayes Classifier, Normal Distribution
(From Unit 3.3 of syllabus)
"""

import numpy as np
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns


class NaiveBayesClassifier:
    """
    Naive Bayes Classifier with different variants
    (From Unit 3.3 of syllabus)
    """
    
    def __init__(self, variant='gaussian'):
        """
        variant: 'gaussian', 'multinomial', or 'bernoulli'
        """
        self.variant = variant
        
        if variant == 'gaussian':
            self.model = GaussianNB()
        elif variant == 'multinomial':
            self.model = MultinomialNB()
        elif variant == 'bernoulli':
            self.model = BernoulliNB()
        else:
            raise ValueError("Variant must be 'gaussian', 'multinomial', or 'bernoulli'")
        
        self.is_trained = False
        self.class_prior_ = None
        self.classes_ = None
    
    def train(self, X_train, y_train):
        """Train Naive Bayes classifier"""
        print(f"\nTraining Naive Bayes Classifier ({self.variant})...")
        
        self.model.fit(X_train, y_train)
        self.is_trained = True
        self.classes_ = self.model.classes_
        self.class_prior_ = self.model.class_prior_
        
        print("Naive Bayes training completed!")
        print(f"Classes: {self.classes_}")
        print(f"Class priors: {self.class_prior_}")
        
        return self.model
    
    def predict(self, X):
        """Make predictions"""
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """
        Get class probability estimates
        (Demonstrates probabilistic classification)
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        return self.model.predict_proba(X)
    
    def predict_log_proba(self, X):
        """Get log probability estimates"""
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        return self.model.predict_log_proba(X)
    
    def get_class_priors(self):
        """Get prior probabilities of classes"""
        if not self.is_trained:
            raise ValueError("Model must be trained first")
        
        print("\nClass Prior Probabilities:")
        for cls, prior in zip(self.classes_, self.class_prior_):
            print(f"Class {cls}: {prior:.4f}")
        
        return dict(zip(self.classes_, self.class_prior_))


class NormalDistributionAnalysis:
    """
    Normal Distribution Analysis and Geometric Interpretations
    (From Unit 3.3 of syllabus)
    """
    
    def __init__(self):
        self.mean = None
        self.std = None
        self.variance = None
    
    def fit(self, data):
        """Fit normal distribution to data"""
        self.mean = np.mean(data)
        self.std = np.std(data)
        self.variance = np.var(data)
        
        print("\nNormal Distribution Parameters:")
        print(f"Mean (μ): {self.mean:.4f}")
        print(f"Standard Deviation (σ): {self.std:.4f}")
        print(f"Variance (σ²): {self.variance:.4f}")
        
        return self.mean, self.std
    
    def probability_density(self, x):
        """Calculate probability density at point x"""
        if self.mean is None or self.std is None:
            raise ValueError("Fit the distribution first")
        
        return stats.norm.pdf(x, loc=self.mean, scale=self.std)
    
    def cumulative_probability(self, x):
        """Calculate cumulative probability P(X <= x)"""
        if self.mean is None or self.std is None:
            raise ValueError("Fit the distribution first")
        
        return stats.norm.cdf(x, loc=self.mean, scale=self.std)
    
    def visualize_distribution(self, data, title="Normal Distribution"):
        """
        Visualize the normal distribution with geometric interpretation
        """
        if self.mean is None or self.std is None:
            self.fit(data)
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Histogram with fitted normal curve
        axes[0].hist(data, bins=30, density=True, alpha=0.7, 
                     color='skyblue', edgecolor='black')
        
        # Plot fitted normal distribution
        x_range = np.linspace(data.min(), data.max(), 1000)
        axes[0].plot(x_range, stats.norm.pdf(x_range, self.mean, self.std),
                    'r-', linewidth=2, label='Fitted Normal Distribution')
        
        axes[0].axvline(self.mean, color='green', linestyle='--', 
                       linewidth=2, label=f'Mean = {self.mean:.2f}')
        axes[0].axvline(self.mean - self.std, color='orange', linestyle='--',
                       linewidth=1.5, alpha=0.7, label=f'±1σ')
        axes[0].axvline(self.mean + self.std, color='orange', linestyle='--',
                       linewidth=1.5, alpha=0.7)
        
        axes[0].set_xlabel('Value', fontsize=12)
        axes[0].set_ylabel('Probability Density', fontsize=12)
        axes[0].set_title(f'{title} - Histogram & Fitted Curve', fontsize=14)
        axes[0].legend()
        axes[0].grid(alpha=0.3)
        
        # Q-Q plot for normality test
        stats.probplot(data, dist="norm", plot=axes[1])
        axes[1].set_title('Q-Q Plot (Normality Test)', fontsize=14)
        axes[1].grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('/home/claude/normal_distribution_analysis.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Normal distribution visualization saved")
    
    def test_normality(self, data):
        """
        Test if data follows normal distribution using Shapiro-Wilk test
        """
        statistic, p_value = stats.shapiro(data)
        
        print("\nNormality Test (Shapiro-Wilk):")
        print(f"Test Statistic: {statistic:.4f}")
        print(f"P-value: {p_value:.4f}")
        
        if p_value > 0.05:
            print("Data appears to follow normal distribution (p > 0.05)")
        else:
            print("Data does NOT follow normal distribution (p <= 0.05)")
        
        return statistic, p_value


class DiscriminantAnalysis:
    """
    Discriminative Learning with Maximum Likelihood
    (From Unit 3.3: Discriminative learning with Maximum likelihood)
    """
    
    def __init__(self, method='linear'):
        """
        method: 'linear' for LDA or 'quadratic' for QDA
        """
        self.method = method
        
        if method == 'linear':
            self.model = LinearDiscriminantAnalysis()
            self.model_name = "Linear Discriminant Analysis (LDA)"
        elif method == 'quadratic':
            self.model = QuadraticDiscriminantAnalysis()
            self.model_name = "Quadratic Discriminant Analysis (QDA)"
        else:
            raise ValueError("Method must be 'linear' or 'quadratic'")
        
        self.is_trained = False
    
    def train(self, X_train, y_train):
        """Train discriminant analysis model"""
        print(f"\nTraining {self.model_name}...")
        
        self.model.fit(X_train, y_train)
        self.is_trained = True
        
        print(f"{self.model_name} training completed!")
        
        if self.method == 'linear':
            print(f"Explained variance ratio: {self.model.explained_variance_ratio_}")
        
        return self.model
    
    def predict(self, X):
        """Make predictions"""
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """Get probability estimates using maximum likelihood"""
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        return self.model.predict_proba(X)
    
    def transform(self, X):
        """
        Transform data to discriminant space (LDA only)
        Useful for dimensionality reduction
        """
        if self.method != 'linear':
            raise ValueError("Transform only available for Linear Discriminant Analysis")
        if not self.is_trained:
            raise ValueError("Model must be trained before transformation")
        
        return self.model.transform(X)


class BayesianInference:
    """
    Bayesian Inference utilities
    Demonstrates Bayes theorem and posterior probability calculation
    """
    
    @staticmethod
    def bayes_theorem(prior, likelihood, evidence):
        """
        Calculate posterior probability using Bayes theorem
        P(H|E) = P(E|H) * P(H) / P(E)
        
        prior: P(H) - prior probability
        likelihood: P(E|H) - likelihood
        evidence: P(E) - evidence/marginal probability
        """
        posterior = (likelihood * prior) / evidence
        return posterior
    
    @staticmethod
    def calculate_evidence(likelihoods, priors):
        """
        Calculate total evidence P(E) = Σ P(E|H_i) * P(H_i)
        
        likelihoods: list of P(E|H_i) for each hypothesis
        priors: list of P(H_i) for each hypothesis
        """
        evidence = sum(l * p for l, p in zip(likelihoods, priors))
        return evidence
    
    @staticmethod
    def posterior_probabilities(likelihoods, priors):
        """
        Calculate posterior probabilities for multiple hypotheses
        
        Returns: list of posterior probabilities
        """
        evidence = BayesianInference.calculate_evidence(likelihoods, priors)
        posteriors = [(l * p) / evidence for l, p in zip(likelihoods, priors)]
        
        print("\nBayesian Inference Results:")
        for i, (prior, likelihood, posterior) in enumerate(zip(priors, likelihoods, posteriors)):
            print(f"Hypothesis {i+1}:")
            print(f"  Prior: {prior:.4f}")
            print(f"  Likelihood: {likelihood:.4f}")
            print(f"  Posterior: {posterior:.4f}")
        
        return posteriors
