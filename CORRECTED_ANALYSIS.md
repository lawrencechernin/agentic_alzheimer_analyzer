# Corrected Analysis: Why AUC Dropped to 0.7409

## The Real Story

I made an error in my previous summary. Here's what actually happened:

### **Three Different Models**

1. **Experiment 4 - Clean Baseline**: 0.7559 AUC
   - Self-report labels only
   - **No ECOG/SP-ECOG features** (removed leakage)
   - 5.9% MCI prevalence

2. **Experiment 9 - Self-Report with Leakage**: 0.7227 AUC  
   - Self-report labels only
   - **With ECOG/SP-ECOG features** (potential leakage)
   - 5.7% MCI prevalence

3. **Experiment 9 - Consensus**: 0.7409 AUC
   - Multi-source consensus labels
   - **With ECOG/SP-ECOG features** (potential leakage)
   - 0.7% MCI prevalence

## Key Insights

### **Consensus vs Self-Report (Same Features)**
- **Consensus**: 0.7409 AUC (+0.0182 improvement)
- **Self-Report**: 0.7227 AUC
- **Conclusion**: Consensus validation works when comparing apples-to-apples

### **Consensus vs Clean Baseline (Different Features)**
- **Consensus**: 0.7409 AUC (-0.0150 drop)
- **Clean Baseline**: 0.7559 AUC  
- **Conclusion**: ECOG/SP-ECOG features hurt performance even with consensus labels

## Why the Drop?

The consensus approach (0.7409) is **worse than the clean baseline** (0.7559) because:

1. **ECOG/SP-ECOG Features Are Noise**: Even with consensus labels, these features hurt performance
2. **Label Quality vs Feature Quality**: Better labels can't overcome bad features
3. **Domain Mismatch**: SP-ECOG measures functional changes over 10 years, while MemTrax measures current performance

## The Real Best Model

**The actual best model is Experiment 4 (0.7559 AUC)**:
- Self-report labels only
- No ECOG/SP-ECOG features (no leakage)
- Clean, honest performance

## Corrected Summary

| Model | AUC | Labels | Features | Notes |
|-------|-----|--------|----------|-------|
| **Best (Clean)** | **0.7559** | **Self-report** | **No ECOG/SP-ECOG** | **✅ No leakage** |
| Consensus | 0.7409 | Multi-source | With ECOG/SP-ECOG | ❌ Features hurt performance |
| Self-Report | 0.7227 | Self-report | With ECOG/SP-ECOG | ❌ Leakage + noise |

## Conclusion

The consensus approach **does work** (+0.0182 AUC improvement), but it can't overcome the damage from ECOG/SP-ECOG features. The real best model is the clean baseline (0.7559 AUC) without any cognitive assessment features.

**Key Learning**: Label quality improvements are limited by feature quality. Bad features can hurt performance even with perfect labels.
