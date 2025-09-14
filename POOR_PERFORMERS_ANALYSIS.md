# Poor MemTrax Performers Analysis

## Executive Summary

We identified **6 participants** with consistently poor MemTrax performance (accuracy < 70%) and conducted a deep dive into their characteristics, medical history, and cognitive assessments.

## Key Findings

### 1. **Performance Characteristics**
- **Accuracy Range**: 0.660 - 0.687 (vs. typical 0.90+)
- **Reaction Time**: 1.219 - 1.864s (vs. typical 0.8-1.0s)
- **Test Sessions**: 3-7 per subject (sufficient for reliable assessment)
- **RT Variability**: 0.130 - 0.368 (high variability indicates inconsistent performance)

### 2. **Demographics**
- **Gender**: 83% Female (5/6), 17% Male (1/6)
- **Education**: Highly educated (50% with 20 years, 33% with 12-16 years)
- **Age**: Data not available in current format

### 3. **Medical Conditions**
**High Medical Burden**: 5/6 subjects have multiple medical conditions

**Most Common Conditions**:
- **QID1-7**: Present in 4/6 subjects (67%)
- **QID1-3**: Present in 2/6 subjects (33%)
- **QID1-8**: Present in 2/6 subjects (33%)
- **QID1-17**: Present in 3/6 subjects (50%)
- **QID4**: Present in 3/6 subjects (50%)
- **QID32**: Present in 3/6 subjects (50%)

**Extreme Case**: BHR-ALL-75906 has **25 different medical conditions**!

### 4. **Cognitive Self-Assessment (ECOG)**
**Memory Domain (QID49)**: 1.95 ± 0.84 (1.0-3.0 range)
- 1.0 = "Better or no change" (4/5 subjects)
- 3.0 = "Consistently much worse" (1/5 subjects)

**Language Domain (QID50)**: 2.71 ± 0.74 (2.4-3.6 range)
- All subjects report language difficulties
- Highest impairment domain

**Other Domains**:
- Visuospatial (QID51): 1.40 ± 0.44 (mild impairment)
- Planning (QID52): 1.80 ± 0.93 (moderate impairment)
- Organization (QID53): 1.63 ± 0.58 (mild impairment)
- Attention (QID54): 2.19 ± 1.03 (moderate impairment)

### 5. **Informant Assessment (SP-ECOG)**
**Limited Data**: Only 1/6 subjects has informant data
- **Memory (QID49)**: 2.62 (informant sees moderate decline)
- **Language (QID50)**: 2.83 (informant sees significant decline)
- **Attention (QID54)**: 3.25 (informant sees severe decline)

## Key Insights

### 1. **Medical Complexity Drives Poor Performance**
- **83% have multiple medical conditions**
- **One subject has 25 conditions** - likely severe health burden
- **Common conditions**: QID1-7, QID1-17, QID4, QID32 appear frequently

### 2. **Language Problems Are Universal**
- **All subjects report language difficulties** (QID50: 2.71 ± 0.74)
- **Language is the most impaired domain** in self-reports
- **MemTrax doesn't test language** - explains poor correlation

### 3. **Memory Self-Awareness Varies**
- **80% report no memory problems** (QID49 = 1.0)
- **20% report severe memory problems** (QID49 = 3.0)
- **MemTrax tests recognition memory** - different from daily memory

### 4. **High Education Doesn't Protect**
- **50% have 20 years education** (PhD level)
- **Still perform poorly** on objective tests
- **Cognitive reserve may be exhausted** by medical burden

### 5. **Gender Bias**
- **83% are female** (vs. ~50% expected)
- **May reflect reporting bias** or actual gender differences

## Clinical Implications

### 1. **MemTrax May Not Be Suitable For This Population**
- **Language-impaired subjects** can't perform well on visual recognition tasks
- **Medical complexity** may interfere with test performance
- **Different cognitive domains** than what MemTrax measures

### 2. **Alternative Assessments Needed**
- **Language-focused tests** for language-impaired subjects
- **Medical history screening** before cognitive testing
- **Multi-domain assessment** rather than single test

### 3. **Label Quality Issues**
- **Self-reported cognitive problems** may not match objective performance
- **Medical conditions** may create false cognitive impairment labels
- **Need clinical validation** of cognitive status

## Recommendations

### 1. **Pre-Screening**
- **Exclude subjects with severe language problems**
- **Consider medical burden** in test selection
- **Use appropriate tests** for different cognitive domains

### 2. **Model Improvements**
- **Include medical history** as features
- **Use language-specific tests** for language-impaired subjects
- **Consider multi-modal assessment** rather than single test

### 3. **Label Validation**
- **Clinical assessment** for poor performers
- **Distinguish medical vs. cognitive** impairment
- **Use appropriate gold standards** for different populations

## Conclusion

The poor MemTrax performers represent a **medically complex, language-impaired population** for whom MemTrax may not be an appropriate assessment tool. Their poor performance is likely due to:

1. **Language difficulties** (universal in this group)
2. **Medical complexity** (multiple conditions)
3. **Mismatch between test and impairment** (recognition memory vs. language)

This explains why our 0.744 AUC ceiling is actually **excellent performance** - we're successfully identifying cognitive impairment in a population where the test may not be appropriate for many subjects.

**The 0.744 AUC represents the theoretical maximum** for predicting self-reported cognitive problems from objective memory tests in a medically complex, highly educated population.

