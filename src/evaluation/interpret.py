# src/evaluation/interpret.py
import os
import logging
import sys
import traceback
import threading
import json
import html
import platform
import psutil
import numpy as np
import torch
import shap
import lime
import matplotlib
import importlib.metadata
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass, field
from datetime import datetime
from collections import defaultdict, Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    __version__ as transformers_version
)
from seqeval.metrics import classification_report
from lime.lime_text import LimeTextExplainer
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Configure system for optimal performance
matplotlib.use('Agg')  # Set non-GUI backend before importing pyplot
import matplotlib.pyplot as plt

# Global configuration
torch.backends.cudnn.benchmark = True
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # Suppress TensorFlow logging

# Fix Windows console encoding
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

@dataclass
class NERConfig:
    """Enhanced configuration for NER analysis with production defaults"""
    model_path: str = "models/fine_tuned/ethiomart_ner"
    data_path: str = "data/labeled"
    interpret_samples: int = 5
    shap_samples: int = 50
    lime_samples: int = 1000
    report_dir: str = "reports"
    cache_dir: str = ".cache"
    max_seq_length: int = 128
    batch_size: int = 16
    max_workers: int = min(4, (os.cpu_count() or 1) // 2)  # Optimal parallel processing
    attention_layer: int = -1  # Default to last layer
    attention_head: int = 0    # Default to first head
    min_shap_evals: int = 100  # Minimum SHAP evaluations
    max_shap_evals: int = 500  # Maximum SHAP evaluations
    shap_timeout: int = 30     # Seconds per SHAP explanation
    fallback_strategy: str = "sample"  # Options: "sample", "skip", "permutation"
    
    version_specs: Dict[str, Dict[str, str]] = field(default_factory=lambda: {
        "transformers": {"min": "4.28.0", "max": "4.99.99", "recommended": "4.28.1"},
        "torch": {"min": "1.12.0", "max": "2.9.99", "recommended": "1.12.1"},
        "shap": {"min": "0.41.0", "max": "0.99.99", "recommended": "0.41.0"},
        "lime": {"min": "0.2.0", "max": "0.99.99", "recommended": "0.2.0"}
    })
    
    label_map: Dict[int, str] = field(default_factory=lambda: {
        0: "O", 1: "B-PRODUCT", 2: "I-PRODUCT", 3: "B-PRICE",
        4: "I-PRICE", 5: "B-PHONE", 6: "I-PHONE", 7: "B-LOC"
    })

class NERLogger:
    """Enhanced structured logging with production-grade configuration"""
    
    @staticmethod
    def configure() -> logging.Logger:
        """Configure logging with UTC timestamps and JSON formatting"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s.%(msecs)03dZ %(name)s %(levelname)s %(message)s',
            datefmt='%Y-%m-%dT%H:%M:%S',
            handlers=[
                logging.FileHandler("ner_analysis.log", encoding='utf-8'),
                logging.StreamHandler(sys.stdout)
            ]
        )
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        
        # Suppress noisy library logs
        for lib in ['transformers', 'shap', 'lime', 'matplotlib', 'plotly']:
            logging.getLogger(lib).setLevel(logging.WARNING)
            
        return logger

logger = NERLogger.configure()

class ModelValidator:
    """Enhanced production-grade model and environment validation"""
    
    @staticmethod
    def validate_versions(config: NERConfig) -> bool:
        """Semantic version validation with compatibility ranges"""
        import torch
        import importlib.metadata
        
        version_info = {
            "transformers": transformers_version,
            "torch": torch.__version__,
        }
        
        # Try to get version for optional packages
        for lib in ["shap", "lime", "seqeval", "plotly"]:
            try:
                version_info[lib] = importlib.metadata.version(lib)
            except:
                version_info[lib] = "0.0.0"
        
        all_valid = True
        for lib, specs in config.version_specs.items():
            current = version_info.get(lib, "0.0.0")
            min_ver = ModelValidator.parse_version(specs["min"])
            max_ver = ModelValidator.parse_version(specs["max"])
            current_ver = ModelValidator.parse_version(current)
            
            if not (min_ver <= current_ver <= max_ver):
                logger.warning(
                    f"Version mismatch for {lib}: "
                    f"Current {current} not in range {specs['min']}-{specs['max']}"
                )
                all_valid = False
        
        # Additional critical checks
        if not torch.cuda.is_available():
            logger.warning("CUDA not available - falling back to CPU")
        
        return all_valid
    
    @staticmethod
    def parse_version(version_str: str) -> Tuple[int, int, int]:
        """Robust semantic version parsing"""
        parts = []
        for part in version_str.split('+')[0].split('.'):
            try:
                parts.append(int(part))
            except ValueError:
                parts.append(0)
        return tuple((parts + [0, 0, 0])[:3])

class NERInterpreter:
    """Production-grade model interpretability with elite optimizations"""
    
    def __init__(self, model, tokenizer, label_map, config):
        self.model = model.eval()
        self.tokenizer = tokenizer
        self.label_map = label_map
        self.config = config
        self.device = model.device
        self._setup_dirs()
        self._setup_explainers()
        self.shap_lock = threading.Lock()  # Thread safety for SHAP
        self.tokenizer_lock = threading.Lock()  # Thread safety for tokenizer
        
    def _setup_dirs(self):
        """Create necessary directories for outputs"""
        Path(self.config.cache_dir).mkdir(exist_ok=True)
        Path(self.config.report_dir).mkdir(exist_ok=True)
        Path("interpretability/shap").mkdir(parents=True, exist_ok=True)
        Path("interpretability/lime").mkdir(parents=True, exist_ok=True)
        Path("interpretability/embeddings").mkdir(parents=True, exist_ok=True)
        Path("interpretability/attention").mkdir(parents=True, exist_ok=True)
    
    def _setup_explainers(self):
        """Initialize explainers with optimized configurations"""
        # SHAP explainer with adaptive configuration
        self.shap_explainer = shap.Explainer(
            self._shap_predict_fn,
            masker=shap.maskers.Text(self.tokenizer),
            output_names=list(self.label_map.values()),
            algorithm='auto',
            max_evals=min(2 * self.config.max_seq_length + 4, self.config.max_shap_evals),
            batch_size=max(1, self.config.batch_size // 2)
        )
        
        # LIME explainer with optimized parameters
        self.lime_explainer = LimeTextExplainer(
            class_names=list(self.label_map.values()),
            bow=False,
            kernel_width=25,
            verbose=False
        )
    
    def _shap_predict_fn(self, texts: List[str]) -> np.ndarray:
        """Thread-safe prediction function with shape guarantees and memory optimization"""
        with self.shap_lock:  # Ensure thread safety
            try:
                # Input validation and preprocessing
                processed_texts = [str(text) if text is not None else "" for text in texts]
                
                # Tokenize with thread-safe tokenizer access
                with self.tokenizer_lock:
                    inputs = self.tokenizer(
                        processed_texts,
                        return_tensors="pt",
                        padding='max_length',
                        truncation=True,
                        max_length=self.config.max_seq_length,
                        return_attention_mask=True
                    ).to(self.device)

                # Model prediction with memory optimization
                with torch.no_grad(), torch.amp.autocast(device_type=self.device.type):
                    outputs = self.model(**inputs)
                    logits = outputs.logits.float()  # Ensure FP32 for SHAP compatibility
                
                # Convert to numpy with shape (batch_size, seq_len, num_classes)
                return logits.detach().cpu().numpy()

            except Exception as e:
                logger.error(f"SHAP prediction failed: {str(e)}")
                raise RuntimeError(f"Prediction error: {str(e)}") from e

    def explain_with_shap(self, text: str) -> Dict[str, Any]:
        """Robust SHAP explanation with automatic fallbacks and caching"""
        try:
            # Validate input
            if not text or not isinstance(text, str):
                raise ValueError("Invalid input text")
            
            # Use cached explanation if available
            cache_key = f"shap_{hash(text)}"
            if cached := self._load_from_cache(cache_key):
                return cached

            # Adaptive explanation strategy based on text length
            text_length = len(text.split())
            max_evals = min(
                max(2 * text_length + 1, self.config.min_shap_evals),
                self.config.max_shap_evals
            )
            
            try:
                # First try with optimized settings
                shap_values = self.shap_explainer([text], max_evals=max_evals)
            except (ValueError, RuntimeError, IndexError) as e:
                if "broadcast" in str(e) or "shape" in str(e) or "index" in str(e):
                    # Fallback strategy based on config
                    if self.config.fallback_strategy == "permutation":
                        logger.warning("Using permutation explainer fallback")
                        explainer = shap.explainers.Permutation(
                            self._shap_predict_fn,
                            masker=shap.maskers.Text(self.tokenizer),
                            max_evals=max_evals
                        )
                        shap_values = explainer([text])
                    elif self.config.fallback_strategy == "sample":
                        logger.warning("Using sampling strategy fallback")
                        return self._explain_with_shap_sampling(text, max_evals)
                    else:
                        logger.warning("Skipping failed explanation")
                        return {
                            "status": "skipped",
                            "reason": f"SHAP failed: {str(e)}",
                            "text_length": text_length
                        }
                else:
                    raise

            # Process and validate results
            if not shap_values or len(shap_values) == 0:
                raise ValueError("Empty SHAP values")
                
            # Save visualization and raw data
            result = {
                "status": "success",
                "data": {
                    "values": self._process_shap_values(shap_values[0]),
                    "plot_path": self._save_shap_visualization(shap_values, text),
                    "meta": {
                        "method": str(self.shap_explainer.__class__.__name__),
                        "text_length": text_length,
                        "num_features": len(shap_values[0].values),
                        "max_evals": max_evals
                    }
                }
            }
            
            self._save_to_cache(cache_key, result)
            return result

        except Exception as e:
            logger.error(f"SHAP explanation failed: {str(e)}")
            return {
                "status": "error",
                "error": str(e),
                "remediation": "Try shorter text or check model compatibility",
                "text_length": len(text.split()) if text else 0
            }

    def _explain_with_shap_sampling(self, text: str, max_evals: int) -> Dict[str, Any]:
        """Alternative SHAP explanation using sampling strategy"""
        try:
            # Split text into chunks if too long
            tokens = text.split()
            if len(tokens) > self.config.max_seq_length:
                chunks = [
                    " ".join(tokens[i:i+self.config.max_seq_length])
                    for i in range(0, len(tokens), self.config.max_seq_length)
                ]
                chunk_results = []
                for chunk in chunks:
                    result = self.explain_with_shap(chunk)
                    if result["status"] == "success":
                        chunk_results.extend(result["data"]["values"])
                
                if chunk_results:
                    return {
                        "status": "success",
                        "data": {
                            "values": chunk_results,
                            "plot_path": None,  # Skip visualization for chunks
                            "meta": {
                                "method": "chunked_SHAP",
                                "text_length": len(tokens),
                                "num_features": len(chunk_results),
                                "max_evals": max_evals
                            }
                        }
                    }
                raise ValueError("All chunks failed")
            
            # If text is short but still failing, try with reduced complexity
            explainer = shap.Explainer(
                self._shap_predict_fn,
                masker=shap.maskers.Text(self.tokenizer),
                output_names=list(self.label_map.values()),
                algorithm='permutation',
                max_evals=max(100, min(max_evals, 200)),  # Conservative range
                batch_size=1  # Most stable setting
            )
            
            shap_values = explainer([text])
            return {
                "status": "success",
                "data": {
                    "values": self._process_shap_values(shap_values[0]),
                    "plot_path": self._save_shap_visualization(shap_values, text),
                    "meta": {
                        "method": "fallback_SHAP",
                        "text_length": len(tokens),
                        "num_features": len(shap_values[0].values),
                        "max_evals": max_evals
                    }
                }
            }
        except Exception as e:
            logger.error(f"SHAP sampling fallback failed: {str(e)}")
            raise

    def _process_shap_values(self, shap_output) -> List[Dict]:
        """Convert SHAP values to interpretable format with alignment checks"""
        features = []
        max_tokens = min(len(shap_output.data), len(shap_output.values))  # Prevent index errors
        
        for i in range(max_tokens):
            token = shap_output.data[i]
            values = shap_output.values[i]
            
            if token in self.tokenizer.all_special_tokens:
                continue
                
            # Validate shape consistency
            if len(values) != len(self.label_map):
                logger.warning(f"Shape mismatch at token {i}")
                continue
                
            features.append({
                "token": token,
                "values": {label: float(values[j]) for j, label in enumerate(self.label_map.values())},
                "importance": np.linalg.norm(values, ord=2)  # L2 norm
            })
        return features

    def _save_shap_visualization(self, shap_values, text: str) -> str:
        """Enhanced SHAP visualization saving with error handling"""
        try:
            viz_dir = Path("interpretability") / "shap"
            viz_dir.mkdir(parents=True, exist_ok=True)
            
            safe_text = "".join(c if c.isalnum() else "_" for c in text[:50])
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            plot_path = viz_dir / f"shap_{safe_text}_{timestamp}.png"
            
            plt.figure(figsize=(12, 6))
            shap.plots.text(shap_values[0])
            plt.title(f"SHAP Explanation - {safe_text}", fontsize=10)
            plt.tight_layout()
            plt.savefig(plot_path, bbox_inches="tight", dpi=150)
            plt.close()
            
            # Save raw data as well
            data_path = viz_dir / f"shap_{safe_text}_{timestamp}.json"
            with open(data_path, "w") as f:
                json.dump({
                    "text": text,
                    "shap_values": [v.tolist() for v in shap_values[0].values],
                    "tokens": shap_values[0].data
                }, f)
            
            return str(plot_path)
        except Exception as e:
            logger.error(f"Failed to save SHAP visualization: {str(e)}")
            return ""

    # [Rest of the class implementations remain similar but with added error handling]
    def explain_with_lime(self, text: str) -> Dict[str, Any]:
        """Optimized LIME explanation with adaptive sampling"""
        try:
            # Use cached explanation if available
            cache_key = f"lime_{hash(text)}"
            if cached := self._load_from_cache(cache_key):
                return cached
                
            # Adaptive sampling based on text length
            text_length = len(text.split())
            num_samples = min(self.config.lime_samples, text_length * 100)
            
            exp = self.lime_explainer.explain_instance(
                text,
                self._lime_predict_fn,
                num_features=10,
                num_samples=num_samples,
                distance_metric='cosine'
            )
            
            result = {
                "status": "success",
                "data": {
                    "explanation": exp.as_list(),
                    "html_path": self._save_lime_explanation(exp, text),
                    "top_features": self._get_lime_top_features(exp),
                    "meta": {
                        "num_samples": num_samples,
                        "num_features": 10,
                        "text_length": text_length
                    }
                }
            }
            
            self._save_to_cache(cache_key, result)
            return result
            
        except Exception as e:
            logger.error(f"LIME explanation failed: {str(e)}")
            return {
                "status": "error",
                "error": str(e),
                "text": str(text)[:100] + "..."
            }

    def _lime_predict_fn(self, texts: List[str]) -> np.ndarray:
        """Thread-safe prediction function for LIME"""
        try:
            # Create a new tokenizer instance for this thread
            local_tokenizer = AutoTokenizer.from_pretrained(self.config.model_path, use_fast=True)
            batch_size = self.config.batch_size
            all_logits = []
            
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i+batch_size]
                
                inputs = local_tokenizer(
                    batch,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=self.config.max_seq_length
                ).to(self.device)
                
                with torch.no_grad():
                    outputs = self.model(**inputs)
                
                all_logits.append(outputs.logits.cpu().numpy())
            
            return np.concatenate(all_logits, axis=0)
            
        except Exception as e:
            logger.error(f"LIME prediction failed: {str(e)}")
            raise

    def _get_lime_top_features(self, exp) -> List[Tuple[str, float]]:
        """Get top features with confidence scores"""
        return sorted(
            [(x[0], float(x[1])) for x in exp.as_list()],
            key=lambda x: abs(x[1]),
            reverse=True
        )[:10]

    def _save_lime_explanation(self, exp, text: str) -> str:
        """Save LIME explanation with enhanced formatting"""
        html_dir = Path("interpretability") / "lime"
        html_dir.mkdir(parents=True, exist_ok=True)
        
        safe_text = "".join(c if c.isalnum() else "_" for c in text[:50])
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        html_path = html_dir / f"lime_{safe_text}_{timestamp}.html"
        
        # Generate enhanced HTML with raw data
        exp.save_to_file(html_path)
        
        # Save raw data as JSON
        data_path = html_dir / f"lime_{safe_text}_{timestamp}.json"
        with open(data_path, "w") as f:
            json.dump({
                "text": text,
                "explanation": exp.as_list(),
                "top_features": self._get_lime_top_features(exp)
            }, f)
        
        return str(html_path)

    def analyze_embeddings(self, samples: List[List[Tuple[str, str]]]) -> Dict[str, Any]:
        """Advanced embedding analysis with automatic dimensionality selection"""
        try:
            # Phase 1: Data Preparation
            embeddings, labels = self._extract_aligned_embeddings(samples[:100])  # Safety cap
            
            # Phase 2: Adaptive Dimensionality Reduction
            pca_results, tsne_results = self._reduce_dimensionality(embeddings)
            
            # Phase 3: Visualization and Analysis
            return {
                "status": "success",
                "data": {
                    "plot_path": self._generate_embedding_visualization(pca_results, tsne_results, labels),
                    "analysis": self._analyze_embedding_clusters(embeddings, labels),
                    "stats": {
                        "num_samples": len(embeddings),
                        "avg_embedding_norm": np.mean(np.linalg.norm(embeddings, axis=1))
                    }
                }
            }
            
        except Exception as e:
            logger.error(f"Embedding analysis failed: {str(e)}")
            return {
                "status": "error",
                "error": str(e),
                "fallback_action": "Reducing sample size or checking token alignment"
            }

    def _extract_aligned_embeddings(self, samples):
        """Precision alignment of embeddings with original tokens"""
        embeddings = []
        labels = []
        
        for sample in samples:
            text = " ".join([t[0] for t in sample])
            sample_labels = [t[1] for t in sample]
            
            # Tokenize with alignment tracking
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                return_offsets_mapping=True,
                truncation=True,
                max_length=self.config.max_seq_length
            ).to(self.device)
            
            # Get embeddings
            with torch.no_grad():
                outputs = self.model(
                    input_ids=inputs['input_ids'],
                    attention_mask=inputs['attention_mask'],
                    output_hidden_states=True
                )
            
            # Align at word level
            word_embeddings = []
            word_ids = inputs.word_ids()
            last_word = None
            
            for idx, word_id in enumerate(word_ids):
                if word_id is None or word_id == last_word:
                    continue
                word_embeddings.append(outputs.hidden_states[-1][0, idx].cpu().numpy())
                last_word = word_id
            
            # Ensure perfect alignment
            min_len = min(len(word_embeddings), len(sample_labels))
            embeddings.extend(word_embeddings[:min_len])
            labels.extend(sample_labels[:min_len])
        
        return np.array(embeddings), labels

    def _reduce_dimensionality(self, embeddings):
        """Intelligent dimensionality reduction with auto-configuration"""
        # Phase 1: PCA with variance-based component selection
        pca = PCA(n_components='mle', random_state=42)  # Automatic component selection
        pca_results = pca.fit_transform(embeddings)
        
        # Phase 2: t-SNE with adaptive perplexity
        perplexity = min(30, max(5, len(embeddings) // 4))
        tsne = TSNE(
            n_components=2,
            perplexity=perplexity,
            early_exaggeration=12,
            learning_rate='auto',
            init='pca',
            random_state=42,
            n_jobs=-1
        )
        tsne_results = tsne.fit_transform(embeddings)
        
        return pca_results, tsne_results

    def _generate_embedding_visualization(self, pca, tsne, labels):
        """Interactive visualization with plotly"""
        fig = make_subplots(rows=1, cols=2, subplot_titles=("PCA", "t-SNE"))
        
        # Add 3D PCA if enough components
        if pca.shape[1] >= 3:
            fig.add_trace(
                go.Scatter3d(
                    x=pca[:, 0], y=pca[:, 1], z=pca[:, 2],
                    mode='markers',
                    marker=dict(size=3, color=[hash(l) for l in labels]),
                    text=labels,
                    name='PCA'
                ),
                row=1, col=1
            )
        else:
            # Fallback to 2D
            fig.add_trace(
                go.Scatter(
                    x=pca[:, 0], y=pca[:, 1],
                    mode='markers',
                    marker=dict(size=5, color=[hash(l) for l in labels]),
                    text=labels,
                    name='PCA'
                ),
                row=1, col=1
            )
        
        # Add t-SNE plot
        fig.add_trace(
            go.Scatter(
                x=tsne[:, 0], y=tsne[:, 1],
                mode='markers',
                marker=dict(size=5, color=[hash(l) for l in labels]),
                text=labels,
                name='t-SNE'
            ),
            row=1, col=2
        )
        
        # Save visualization
        viz_path = Path("interpretability/embeddings") / f"embeddings_{datetime.now().timestamp()}.html"
        fig.write_html(
            viz_path,
            config={'responsive': True},
            include_plotlyjs='cdn',
            full_html=False
        )
        
        return str(viz_path)

    def _analyze_embedding_clusters(self, embeddings, labels):
        """Advanced cluster analysis using HDBSCAN"""
        try:
            import hdbscan
            clusterer = hdbscan.HDBSCAN(
                min_cluster_size=5,
                min_samples=3,
                cluster_selection_epsilon=0.5,
                prediction_data=True
            )
            clusters = clusterer.fit_predict(embeddings)
            
            return {
                "cluster_quality": float(clusterer.relative_validity_),
                "num_clusters": len(set(clusters)) - (1 if -1 in clusters else 0),
                "label_distribution": {
                    str(cluster): Counter(np.array(labels)[clusters == cluster])
                    for cluster in set(clusters)
                }
            }
        except ImportError:
            return {"status": "hdbscan_not_available"}

    def _analyze_attention(self, text: str) -> Dict[str, Any]:
        """Enhanced attention pattern analysis"""
        try:
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                return_offsets_mapping=True
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(**inputs, output_attentions=True)
            
            # Process attention from all layers and heads
            attentions = torch.stack(outputs.attentions).cpu().numpy()  # [layers, heads, seq_len, seq_len]
            
            # Create enhanced visualization
            plot_path = self._save_attention_plot(
                attentions,
                self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
            )
            
            # Calculate attention statistics
            selected_layer = self.config.attention_layer if 0 <= self.config.attention_layer < attentions.shape[0] else -1
            selected_head = self.config.attention_head if 0 <= self.config.attention_head < attentions.shape[1] else 0
            
            attention_matrix = attentions[selected_layer, selected_head]
            max_attention = np.max(attention_matrix)
            min_attention = np.min(attention_matrix)
            avg_attention = np.mean(attention_matrix)
            
            return {
                "status": "success",
                "data": {
                    "plot_path": plot_path,
                    "num_layers": attentions.shape[0],
                    "num_heads": attentions.shape[1],
                    "attention_stats": {
                        "max": float(max_attention),
                        "min": float(min_attention),
                        "mean": float(avg_attention),
                        "selected_layer": selected_layer,
                        "selected_head": selected_head
                    }
                }
            }
            
        except Exception as e:
            logger.error(f"Attention analysis failed: {str(e)}")
            return {
                "status": "error",
                "error": str(e)
            }

    def _save_attention_plot(self, attentions: np.ndarray, tokens: List[str]) -> str:
        """Enhanced attention visualization"""
        # Select representative layers/heads
        num_layers, num_heads = attentions.shape[:2]
        layer_to_show = self.config.attention_layer if 0 <= self.config.attention_layer < num_layers else num_layers // 2
        head_to_show = self.config.attention_head if 0 <= self.config.attention_head < num_heads else num_heads // 2
        
        attention_matrix = attentions[layer_to_show, head_to_show]
        
        # Create enhanced plot
        fig = px.imshow(
            attention_matrix,
            x=tokens,
            y=tokens,
            labels={'x': 'Key', 'y': 'Query'},
            title=f"Attention Layer {layer_to_show} Head {head_to_show}",
            color_continuous_scale='Viridis',
            aspect="auto"
        )
        
        # Save plot with metadata
        viz_dir = Path("interpretability") / "attention"
        viz_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_path = viz_dir / f"attention_{timestamp}.html"
        
        fig.write_html(plot_path, include_plotlyjs='cdn')
        return str(plot_path)

    # Helper methods...
    def _load_from_cache(self, key: str) -> Optional[Dict]:
        """Enhanced cache loading with validation"""
        cache_path = Path(self.config.cache_dir) / f"{key}.json"
        if cache_path.exists():
            try:
                with open(cache_path, "r") as f:
                    data = json.load(f)
                    # Validate cache integrity
                    if isinstance(data, dict) and "status" in data:
                        return data
            except Exception as e:
                logger.warning(f"Cache corrupted for {key}: {str(e)}")
                return None
        return None

    def _save_to_cache(self, key: str, data: Dict):
        """Enhanced cache saving with atomic write"""
        cache_path = Path(self.config.cache_dir) / f"{key}.json"
        temp_path = cache_path.with_suffix(".tmp")
        try:
            with open(temp_path, "w") as f:
                json.dump(data, f)
            temp_path.replace(cache_path)  # Atomic operation
        except Exception as e:
            logger.warning(f"Failed to cache explanation: {str(e)}")
            if temp_path.exists():
                temp_path.unlink()       
        
    
class NERAnalyzer:
    """Elite-level NER analysis with all optimizations"""
    def __init__(self, config: NERConfig):
        self.config = config
        self.model_path = Path(config.model_path)
        self.data_path = Path(config.data_path)
        self.tokenizer = None
        self.model = None
        self.interpreter = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Validate environment
        if not ModelValidator.validate_versions(config):
            logger.warning("Version mismatches detected - proceeding with caution")
    
    def load_model(self) -> bool:
        """Enhanced model loading with validation"""
        try:
            logger.info(f"Loading model from {self.model_path}")
            
            # Load with error handling for corrupted files
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.model_path,
                    use_fast=True
                )
                self.model = AutoModelForTokenClassification.from_pretrained(
                    self.model_path,
                    num_labels=len(self.config.label_map)
                ).to(self.device)
            except Exception as e:
                logger.error(f"Initial model loading failed: {str(e)}")
                # Try with local_files_only as fallback
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.model_path,
                    use_fast=True,
                    local_files_only=True
                )
                self.model = AutoModelForTokenClassification.from_pretrained(
                    self.model_path,
                    num_labels=len(self.config.label_map),
                    local_files_only=True
                ).to(self.device)
            
            # Initialize interpreter
            self.interpreter = NERInterpreter(
                self.model,
                self.tokenizer,
                self.config.label_map,
                self.config
            )
            
            # Verify label alignment
            if self.model.config.id2label != self.config.label_map:
                logger.warning("Updating label map to match model configuration")
                self.config.label_map = self.model.config.id2label
            
            # Enhanced warm-up run
            try:
                with torch.no_grad():
                    test_input = torch.randint(
                        low=0, 
                        high=len(self.tokenizer),
                        size=(1, 10),
                        dtype=torch.long
                    ).to(self.device)
                    _ = self.model(test_input)
                    logger.info("Model warm-up successful")
            except Exception as e:
                logger.error(f"Model warm-up failed: {str(e)}")
                return False
            
            return True
        except Exception as e:
            logger.error(f"Model loading failed: {str(e)}")
            return False

    # [Rest of the NERAnalyzer class implementations with enhanced error handling]
    def evaluate(self) -> Dict[str, Any]:
        """Elite-level evaluation pipeline with all optimizations"""
        try:
            # Load validation data with progress tracking
            logger.info("Loading validation data")
            samples = self._load_samples()
            
            if not samples:
                raise ValueError("No validation samples loaded")
            
            # Standard evaluation metrics with timing
            logger.info("Calculating metrics")
            metrics = self._calculate_metrics(samples)
            
            if not metrics:
                raise ValueError("Metrics calculation failed")
            
            # Enhanced parallel analyses
            logger.info("Running interpretability analyses")
            with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
                futures = {
                    executor.submit(self._analyze_failure_cases, samples): "failure_cases",
                    executor.submit(self._analyze_feature_importance, samples): "feature_importance",
                    executor.submit(self.interpreter.analyze_embeddings, samples): "embedding_analysis"
                }
                
                interpretability = {}
                for future in tqdm(as_completed(futures), total=len(futures), desc="Analyzing"):
                    key = futures[future]
                    interpretability[key] = future.result()
            
            # Generate comprehensive report
            logger.info("Generating report")
            report = {
                "metadata": self._get_metadata(),
                "metrics": metrics,
                "interpretability": interpretability,
                "recommendations": self._generate_recommendations(metrics, interpretability)
            }
            
            self._save_report(report)
            return report
            
        except Exception as e:
            logger.error(f"Evaluation failed: {str(e)}")
            return {
                "error": str(e),
                "traceback": traceback.format_exc()
            }

    def _load_samples(self) -> List[List[Tuple[str, str]]]:
        """Enhanced sample loading with validation"""
        samples = []
        current_sample = []
        
        try:
            # Try to find the validation file with different possible extensions
            val_files = [
                self.data_path / "val.conll",
                self.data_path / "val.txt",
                self.data_path / "validation.conll",
                self.data_path / "validation.txt"
            ]
            
            val_file = None
            for f in val_files:
                if f.exists():
                    val_file = f
                    break
                    
            if val_file is None:
                raise FileNotFoundError(f"No validation file found in {self.data_path}")
                
            logger.info(f"Loading samples from {val_file}")
            
            with open(val_file, "r", encoding="utf-8") as f:
                for line in tqdm(f, desc="Loading samples"):
                    line = line.strip()
                    if not line:
                        if current_sample:
                            samples.append(current_sample)
                            current_sample = []
                        continue
                    
                    parts = line.split()
                    if len(parts) >= 2:
                        current_sample.append((parts[0], parts[-1]))
                        
            # Add the last sample if file doesn't end with newline
            if current_sample:
                samples.append(current_sample)
                
            logger.info(f"Loaded {len(samples)} samples")
            return samples
            
        except Exception as e:
            logger.error(f"Failed to load samples: {str(e)}")
            raise

    def _calculate_metrics(self, samples: List[List[Tuple[str, str]]]) -> Dict:
        """Enhanced metrics calculation with proper alignment"""
        try:
            sentences, true_labels = [], []
            
            # Convert samples to sentences and labels
            for sample in tqdm(samples, desc="Preparing samples"):
                sentences.append([t[0] for t in sample])
                true_labels.append([t[1] for t in sample])
            
            # Get predictions in batches
            pred_labels = []
            for i in tqdm(range(0, len(sentences), self.config.batch_size), desc="Predicting"):
                batch = sentences[i:i+self.config.batch_size]
                
                inputs = self.tokenizer(
                    batch,
                    is_split_into_words=True,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=self.config.max_seq_length
                ).to(self.device)
                
                with torch.no_grad():
                    outputs = self.model(**inputs)
                
                # Process batch predictions
                batch_preds = torch.argmax(outputs.logits, dim=-1).cpu().numpy()
                
                for j, preds in enumerate(batch_preds):
                    # Align predictions with words
                    word_ids = inputs.word_ids(j)
                    aligned_preds = []
                    current_word = None
                    
                    for k, word_id in enumerate(word_ids):
                        if word_id is None:
                            continue
                        if word_id != current_word:
                            aligned_preds.append(self.config.label_map[preds[k]])
                            current_word = word_id
                    
                    pred_labels.append(aligned_preds)
            
            # Trim to same length
            true_trimmed, pred_trimmed = [], []
            for true, pred in zip(true_labels, pred_labels):
                min_len = min(len(true), len(pred))
                true_trimmed.append(true[:min_len])
                pred_trimmed.append(pred[:min_len])
            
            # Calculate enhanced metrics
            report = classification_report(true_trimmed, pred_trimmed, output_dict=True)
            
            # Add additional statistics
            report["num_samples"] = len(samples)
            report["average_length"] = np.mean([len(s) for s in sentences])
            
            # Calculate class distribution
            class_counts = defaultdict(int)
            for sample in true_labels:
                for label in sample:
                    class_counts[label] += 1
            report["class_distribution"] = dict(class_counts)
            
            return report
            
        except Exception as e:
            logger.error(f"Metrics calculation failed: {str(e)}")
            raise

    def _analyze_failure_cases(self, samples: List[List[Tuple[str, str]]]) -> List[Dict[str, Any]]:
        """Enhanced failure case analysis with parallel processing"""
        results = []
        selected_samples = samples[:self.config.interpret_samples]
        
        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            futures = []
            for sample in selected_samples:
                futures.append(executor.submit(self._analyze_single_case, sample))
            
            for future in tqdm(as_completed(futures), total=len(futures), desc="Analyzing cases"):
                results.append(future.result())
                
        return results
    
    def _analyze_single_case(self, sample: List[Tuple[str, str]]) -> Dict[str, Any]:
        try:
            text = " ".join([t[0] for t in sample])
            true_labels = [t[1] for t in sample] 
            
            # Create a new tokenizer instance for this thread
            local_tokenizer = AutoTokenizer.from_pretrained(self.config.model_path, use_fast=True)
            
            inputs = local_tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=self.config.max_seq_length
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            # Align predictions
            preds = torch.argmax(outputs.logits, dim=-1)[0].tolist()
            word_ids = inputs.word_ids()
            aligned_preds = []
            current_word = None
            for i, word_id in enumerate(word_ids):
                if word_id is None:
                    continue
                if word_id != current_word:
                    aligned_preds.append(self.config.label_map[preds[i]])
                    current_word = word_id
            
            # Identify errors
            errors = []
            for idx, (true, pred) in enumerate(zip(true_labels[:len(aligned_preds)], aligned_preds)):
                if true != pred and true != "O":
                    errors.append({
                        "token": sample[idx][0],
                        "position": idx,
                        "true": true,
                        "pred": pred,
                        "context": " ".join([
                            sample[max(0, idx-2)][0],
                            sample[max(0, idx-1)][0],
                            sample[idx][0],
                            sample[min(len(sample)-1, idx+1)][0],
                            sample[min(len(sample)-1, idx+2)][0]
                        ])
                    })
            
            # Generate explanations if errors exist
            explanations = {}
            if errors:
                explanations = {
                    "shap": self.interpreter.explain_with_shap(text),
                    "lime": self.interpreter.explain_with_lime(text),
                    "attention": self._analyze_attention(text)
                }
            
            return {
                "text": text,
                "errors": errors,
                "explanations": explanations,
                "num_tokens": len(sample),
                "num_errors": len(errors)
            }
            
        except Exception as e:
            logger.error(f"Failed to analyze case: {str(e)}")
            return {
                "text": " ".join([t[0] for t in sample]),
                "error": str(e)
            }

    def _analyze_feature_importance(self, samples: List[List[Tuple[str, str]]]) -> Dict[str, Any]:
        """Enhanced feature importance analysis with parallel processing"""
        try:
            # Sample diverse examples for analysis
            analysis_samples = [
                sample for sample in samples 
                if any(label != "O" for _, label in sample)
            ][:self.config.shap_samples]
            
            if not analysis_samples:
                return {"status": "skipped", "reason": "No non-O samples found"}
            
            # Parallel SHAP computation
            with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
                futures = []
                for sample in analysis_samples:
                    text = " ".join([t[0] for t in sample])
                    futures.append(executor.submit(self.interpreter.explain_with_shap, text))
                
                all_shap_values = []
                for future in tqdm(as_completed(futures), total=len(futures), desc="Computing SHAP"):
                    result = future.result()
                    if result["status"] == "success":
                        all_shap_values.append(result["data"]["values"])
            
            if not all_shap_values:
                return {"status": "error", "error": "No successful SHAP explanations"}
            
            # Enhanced importance calculation
            label_importance = defaultdict(list)
            token_importance = defaultdict(list)
            
            for shap_values in all_shap_values:
                for token_info in shap_values:
                    token = token_info["token"]
                    for label, value in token_info["values"].items():
                        label_importance[label].append(abs(value))
                        token_importance[(token, label)].append(abs(value))
            
            avg_label_importance = {
                label: {
                    "mean": np.mean(values),
                    "std": np.std(values),
                    "count": len(values)
                }
                for label, values in label_importance.items()
            }
            
            top_tokens = sorted(
                [(k[0], k[1], np.mean(v)) for k, v in token_importance.items()],
                key=lambda x: x[2],
                reverse=True
            )[:20]  # Top 20 most important token-label pairs
            
            return {
                "status": "success",
                "data": {
                    "label_importance": avg_label_importance,
                    "top_tokens": top_tokens,
                    "num_samples": len(all_shap_values)
                }
            }
            
        except Exception as e:
            logger.error(f"Feature importance analysis failed: {str(e)}")
            return {
                "status": "error",
                "error": str(e)
            }

    def _analyze_attention(self, text: str) -> Dict[str, Any]:
        """Delegate to interpreter's attention analysis"""
        return self.interpreter._analyze_attention(text)

    def _generate_recommendations(self, metrics: Dict, interpretability: Dict) -> str:
        """Enhanced actionable recommendations"""
        recommendations = []
        
        # Performance-based recommendations
        f1_score = metrics.get('weighted avg', {}).get('f1-score', 0)
        if f1_score < 0.7:
            rec = f"Model performance is below target (F1 = {f1_score:.3f}). Consider:\n"
            rec += "- Adding more training data, especially for low-performing classes\n"
            
            # Identify weakest classes
            weak_classes = sorted(
                [(k, v) for k, v in metrics.items() if isinstance(v, dict)],
                key=lambda x: x[1].get('f1-score', 0)
            )[:3]
            
            if weak_classes:
                rec += "  Most problematic classes:\n"
                for cls, scores in weak_classes:
                    rec += f"  - {cls}: P={scores.get('precision', 0):.2f}, R={scores.get('recall', 0):.2f}, F1={scores.get('f1-score', 0):.2f}\n"
            
            rec += "- Reviewing label consistency in training data\n"
            rec += "- Trying different hyperparameters or model architectures\n"
            rec += "- Adding contextual features or domain-specific pretraining"
            recommendations.append(rec)
        
        # Error analysis recommendations
        if interpretability.get('failure_cases'):
            error_types = defaultdict(int)
            for case in interpretability['failure_cases']:
                for error in case.get('errors', []):
                    key = f"{error['true']}→{error['pred']}"
                    error_types[key] += 1
            
            common_errors = sorted(error_types.items(), key=lambda x: x[1], reverse=True)[:3]
            if common_errors:
                rec = "Most common error types:\n"
                for error, count in common_errors:
                    rec += f"- {error}: {count} occurrences\n"
                
                # Add specific suggestions based on error types
                for error in common_errors:
                    if "O→" in error[0]:
                        rec += f"  For {error[0]}, consider adding more examples of this entity type\n"
                    elif "→O" in error[0]:
                        rec += f"  For {error[0]}, check for ambiguous contexts causing false negatives\n"
                    else:
                        rec += f"  For {error[0]}, review annotation guidelines for these similar types\n"
                
                recommendations.append(rec)
        
        # Feature importance recommendations
        feat_imp = interpretability.get('feature_importance', {}).get('data', {})
        if feat_imp.get('label_importance'):
            important_labels = sorted(
                [(k, v['mean']) for k, v in feat_imp['label_importance'].items()],
                key=lambda x: x[1],
                reverse=True
            )[:3]
            
            if important_labels:
                rec = "Most important features by label:\n"
                for label, importance in important_labels:
                    rec += f"- {label}: μ={importance:.3f}, σ={feat_imp['label_importance'][label]['std']:.3f}\n"
                
                # Add specific suggestions
                top_tokens = feat_imp.get('top_tokens', [])[:5]
                if top_tokens:
                    rec += "\nMost predictive tokens:\n"
                    for token, label, imp in top_tokens:
                        rec += f"- '{token}' for {label}: {imp:.3f}\n"
                
                recommendations.append(rec)
        
        # Embedding analysis recommendations
        if interpretability.get('embedding_analysis', {}).get('status') == "success":
            embed_data = interpretability['embedding_analysis']['data']
            if embed_data.get('pca_variance', [0,0,0])[0] < 0.5:
                recommendations.append(
                    "PCA variance is low (<50%) in first component - consider:\n"
                    "- Using different embedding strategies\n"
                    "- Adding dimensionality reduction techniques"
                )
        
        return "\n\n".join(recommendations) if recommendations else "No specific recommendations - model looks good!"

    def _get_metadata(self) -> Dict[str, Any]:
        """Enhanced system and model metadata"""
        return {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "system": {
                "platform": platform.platform(),
                "processor": platform.processor(),
                "memory_gb": psutil.virtual_memory().total / (1024 ** 3),
                "python_version": platform.python_version(),
                "cpu_cores": psutil.cpu_count(),
                "load_avg": os.getloadavg() if hasattr(os, 'getloadavg') else None
            },
            "gpu": self._get_gpu_info(),
            "model": {
                "path": str(self.model_path),
                "num_parameters": sum(p.numel() for p in self.model.parameters()),
                "trainable_parameters": sum(p.numel() for p in self.model.parameters() if p.requires_grad),
                "config": self.model.config.to_dict(),
                "tokenizer": {
                    "vocab_size": self.tokenizer.vocab_size,
                    "model_max_length": self.tokenizer.model_max_length
                }
            },
            "environment": {
                "torch_version": torch.__version__,
                "transformers_version": transformers_version,
                "shap_version": importlib.metadata.version("shap") if importlib.metadata.version("shap") else None,
                "lime_version": importlib.metadata.version("lime") if importlib.metadata.version("lime") else None
            }
        }

    def _get_gpu_info(self) -> Dict[str, Any]:
        """Enhanced GPU information"""
        if not torch.cuda.is_available():
            return {"available": False}
        
        info = {
            "available": True,
            "count": torch.cuda.device_count(),
            "current_device": torch.cuda.current_device(),
            "device_name": torch.cuda.get_device_name(),
            "memory_allocated_gb": torch.cuda.memory_allocated() / (1024 ** 3),
            "memory_reserved_gb": torch.cuda.memory_reserved() / (1024 ** 3),
            "cuda_capability": torch.cuda.get_device_capability(),
            "driver_version": torch.cuda.get_driver_version()
        }
        
        # Add per-device details
        info["devices"] = []
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            info["devices"].append({
                "name": props.name,
                "total_memory_gb": props.total_memory / (1024 ** 3),
                "multi_processor_count": props.multi_processor_count,
                "major": props.major,
                "minor": props.minor
            })
        
        return info

    def _save_report(self, report: Dict[str, Any]):
        """Enhanced comprehensive HTML report generation"""
        try:
            report_dir = Path(self.config.report_dir)
            report_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_path = report_dir / f"ner_report_{timestamp}.html"
            
            # Format report content
            html_content = self._format_html_report(report)
            
            # Write to file with atomic operation
            temp_path = report_path.with_suffix(".tmp")
            with open(temp_path, "w", encoding="utf-8") as f:
                f.write(html_content)
            temp_path.replace(report_path)
            
            logger.info(f"Report saved to {report_path}")
            
            # Save raw JSON data with proper type conversion
            json_path = report_dir / f"ner_report_{timestamp}.json"
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2, default=self._convert_for_json)
                
        except Exception as e:
            logger.error(f"Failed to save report: {str(e)}")
            raise

    @staticmethod
    def _convert_for_json(o):
        """Convert numpy types to native Python types for JSON serialization"""
        if isinstance(o, (np.int32, np.int64, np.int_)):
            return int(o)
        if isinstance(o, (np.float32, np.float64, np.float_)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, np.bool_):
            return bool(o)
        raise TypeError(f"Object of type {o.__class__.__name__} is not JSON serializable")
        
    def _format_html_report(self, report: Dict[str, Any]) -> str:
        """Enhanced HTML content generation"""
        metrics = report.get('metrics', {})
        interpretability = report.get('interpretability', {})
        recommendations = report.get('recommendations', "No recommendations generated")
        metadata = report.get('metadata', {})
        
        # Format metrics section
        metrics_html = """
        <div class="metrics">
            <h2>Performance Metrics</h2>
            <table>
                <tr><th>Metric</th><th>Value</th></tr>
        """
        
        if isinstance(metrics, dict) and 'weighted avg' in metrics:
            metrics_html += f"""
                <tr><td>Overall F1</td><td>{metrics['weighted avg']['f1-score']:.4f}</td></tr>
                <tr><td>Precision</td><td>{metrics['weighted avg']['precision']:.4f}</td></tr>
                <tr><td>Recall</td><td>{metrics['weighted avg']['recall']:.4f}</td></tr>
                <tr><td>Number of Samples</td><td>{metrics.get('num_samples', 'N/A')}</td></tr>
                <tr><td>Average Length</td><td>{metrics.get('average_length', 'N/A'):.1f}</td></tr>
            """
            
            # Add per-class metrics if available
            if any(k not in ['micro avg', 'macro avg', 'weighted avg'] for k in metrics):
                metrics_html += """
                <tr><td colspan="2"><h3>Per-class Metrics</h3></td></tr>
                """
                for cls, scores in metrics.items():
                    if isinstance(scores, dict):
                        metrics_html += f"""
                        <tr>
                            <td>{cls}</td>
                            <td>P={scores.get('precision', 0):.3f}, 
                                R={scores.get('recall', 0):.3f}, 
                                F1={scores.get('f1-score', 0):.3f}, 
                                Support={scores.get('support', 0)}</td>
                        </tr>
                        """
        else:
            metrics_html += "<tr><td colspan='2'>No metrics available</td></tr>"
        
        metrics_html += "</table></div>"
        
        # Format class distribution
        if metrics.get('class_distribution'):
            dist_html = """
            <div class="distribution">
                <h3>Class Distribution</h3>
                <table>
                    <tr><th>Class</th><th>Count</th><th>Percentage</th></tr>
            """
            total = sum(metrics['class_distribution'].values())
            for cls, count in sorted(metrics['class_distribution'].items()):
                dist_html += f"""
                <tr>
                    <td>{cls}</td>
                    <td>{count}</td>
                    <td>{count/total*100:.1f}%</td>
                </tr>
                """
            dist_html += "</table></div>"
            metrics_html += dist_html
        
        # Format failure cases
        cases_html = []
        for case in interpretability.get('failure_cases', []):
            errors = "<br>".join(
                f"Token '{err['token']}' (pos {err['position']}): Expected {err['true']}, Predicted {err['pred']}<br>"
                f"Context: {err.get('context', '')}"
                for err in case.get('errors', [])
            )
            
            # Add visualizations if available
            viz_html = ""
            explanations = case.get('explanations', {})
            
            if explanations.get('shap', {}).get('status') == "success":
                shap_plot = explanations['shap']['data']['plot_path']
                viz_html += f"""
                <h4>SHAP Explanation</h4>
                <img src="{shap_plot}" width="800">
                <p><small>Max evaluations: {explanations['shap']['data']['meta']['max_evals']}</small></p>
                """
                
            if explanations.get('lime', {}).get('status') == "success":
                lime_html = explanations['lime']['data']['html_path']
                viz_html += f"""
                <h4>LIME Explanation</h4>
                <iframe src="{lime_html}" width="800" height="400"></iframe>
                <p><small>Samples: {explanations['lime']['data']['meta']['num_samples']}</small></p>
                """
            
            if explanations.get('attention', {}).get('status') == "success":
                attention_plot = explanations['attention']['data']['plot_path']
                viz_html += f"""
                <h4>Attention Visualization</h4>
                <iframe src="{attention_plot}" width="800" height="600"></iframe>
                <p><small>Layer {explanations['attention']['data']['attention_stats']['selected_layer']}, 
                Head {explanations['attention']['data']['attention_stats']['selected_head']}</small></p>
                """
            
            cases_html.append(f"""
            <div class="case">
                <h3>Text: {html.escape(case.get('text', ''))}</h3>
                <p><strong>Tokens:</strong> {case.get('num_tokens', 'N/A')} | 
                <strong>Errors:</strong> {case.get('num_errors', 0)}</p>
                <div class="errors"><pre>{html.escape(errors)}</pre></div>
                {viz_html}
            </div>
            """)
        
        # Format feature importance
        feat_imp_html = ""
        if interpretability.get('feature_importance', {}).get('status') == "success":
            feat_data = interpretability['feature_importance']['data']
            feat_imp_html = """
            <div class="feature-importance">
                <h2>Feature Importance Analysis</h2>
                <p>Based on {feat_data['num_samples']} samples</p>
                <h3>Label Importance</h3>
                <table>
                    <tr><th>Label</th><th>Mean Importance</th><th>Std Dev</th><th>Count</th></tr>
            """
            for label, imp in sorted(feat_data['label_importance'].items(), key=lambda x: x[1]['mean'], reverse=True):
                feat_imp_html += f"""
                <tr>
                    <td>{label}</td>
                    <td>{imp['mean']:.4f}</td>
                    <td>{imp['std']:.4f}</td>
                    <td>{imp['count']}</td>
                </tr>
                """
            feat_imp_html += "</table>"
            
            if feat_data.get('top_tokens'):
                feat_imp_html += """
                <h3>Top Predictive Tokens</h3>
                <table>
                    <tr><th>Token</th><th>Label</th><th>Importance</th></tr>
                """
                for token, label, imp in feat_data['top_tokens']:
                    feat_imp_html += f"""
                    <tr>
                        <td>{token}</td>
                        <td>{label}</td>
                        <td>{imp:.4f}</td>
                    </tr>
                    """
                feat_imp_html += "</table>"
            feat_imp_html += "</div>"
        
        # Format embedding analysis
        embedding_html = ""
        if interpretability.get('embedding_analysis', {}).get('status') == "success":
            embed_data = interpretability['embedding_analysis']['data']
            embedding_html = f"""
            <div class="embedding-analysis">
                <h2>Embedding Analysis</h2>
                <p>Analyzed {embed_data['num_embeddings']} token embeddings</p>
                <p>PCA Variance: {', '.join(f'{v:.3f}' for v in embed_data['pca_variance'])}</p>
                <p>t-SNE KL Divergence: {embed_data['tsne_kl_divergence']:.4f}</p>
                <iframe src="{embed_data['plot_path']}" width="1000" height="800"></iframe>
            </div>
            """
        
        # Format metadata
        meta_html = """
        <div class="metadata">
            <h2>System Information</h2>
            <table>
        """
        if metadata.get('system'):
            meta_html += f"""
                <tr><td>Platform</td><td>{metadata['system']['platform']}</td></tr>
                <tr><td>Processor</td><td>{metadata['system']['processor']}</td></tr>
                <tr><td>Memory</td><td>{metadata['system']['memory_gb']:.1f} GB</td></tr>
                <tr><td>Python</td><td>{metadata['system']['python_version']}</td></tr>
            """
        if metadata.get('gpu', {}).get('available'):
            meta_html += f"""
                <tr><td>GPU</td><td>{metadata['gpu']['device_name']}</td></tr>
                <tr><td>CUDA Memory</td><td>{metadata['gpu']['memory_allocated_gb']:.2f} GB allocated</td></tr>
            """
        meta_html += "</table>"
        
        if metadata.get('model'):
            meta_html += f"""
            <h3>Model Information</h3>
            <table>
                <tr><td>Path</td><td>{metadata['model']['path']}</td></tr>
                <tr><td>Parameters</td><td>{metadata['model']['num_parameters']:,}</td></tr>
                <tr><td>Trainable Parameters</td><td>{metadata['model']['trainable_parameters']:,}</td></tr>
                <tr><td>Vocabulary Size</td><td>{metadata['model']['tokenizer']['vocab_size']}</td></tr>
            </table>
            """
        meta_html += "</div>"
        
        # Combine all sections
        return f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>NER Model Analysis Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; line-height: 1.6; }}
                h1, h2, h3, h4 {{ color: #2c3e50; }}
                .section {{ margin-bottom: 30px; border-bottom: 1px solid #eee; padding-bottom: 20px; }}
                .case {{ border: 1px solid #ddd; padding: 15px; margin-bottom: 20px; background: #f9f9f9; }}
                table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background: #f2f2f2; }}
                pre {{ background: #f5f5f5; padding: 10px; overflow-x: auto; }}
                img, iframe {{ max-width: 100%; margin: 10px 0; border: 1px solid #ddd; }}
                .metrics th, .metrics td {{ width: 50%; }}
                .distribution th, .distribution td {{ width: 33%; }}
                .feature-importance th, .feature-importance td {{ width: 25%; }}
                small {{ color: #666; }}
            </style>
        </head>
        <body>
            <h1>NER Model Analysis Report</h1>
            <p>Generated at {metadata.get('timestamp', 'unknown')}</p>
            
            <div class="section">
                <h2>Summary</h2>
                <p>Overall F1-score: <strong>{metrics.get('weighted avg', {}).get('f1-score', 0):.3f}</strong></p>
                <p>Failure cases analyzed: <strong>{len(interpretability.get('failure_cases', []))}</strong></p>
            </div>
            
            <div class="section">
                {metrics_html}
            </div>
            
            <div class="section">
                <h2>Failure Case Analysis</h2>
                {"".join(cases_html)}
            </div>
            
            {feat_imp_html}
            
            {embedding_html}
            
            <div class="section">
                <h2>Recommendations</h2>
                <pre>{html.escape(recommendations)}</pre>
            </div>
            
            {meta_html}
        </body>
        </html>
        """

def main():
    """Production-grade main execution flow"""
    try:
        config = NERConfig()
        analyzer = NERAnalyzer(config)
        
        logger.info("Starting NER analysis")
        
        if not analyzer.load_model():
            raise RuntimeError("Failed to load model")
        
        report = analyzer.evaluate()
        
        if "error" in report:
            raise RuntimeError(f"Evaluation failed: {report['error']}")
        
        print("\nEvaluation Complete")
        print(f"Overall F1: {report['metrics']['weighted avg']['f1-score']:.3f}")
        print(f"Failure cases analyzed: {len(report['interpretability']['failure_cases'])}")
        print(f"Report generated: {config.report_dir}/ner_report_*.html")
        
    except Exception as e:
        logger.error(f"Analysis failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()