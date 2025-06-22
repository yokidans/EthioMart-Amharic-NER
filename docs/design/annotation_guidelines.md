# Annotation Rules
1. **Product**:  
   - `B-PRODUCT`: First word (ስልክ) in "ስልክ 5000 ብር"  
   - `I-PRODUCT`: Subsequent words (if multi-word)

2. **Price**:  
   - `B-PRICE`: Numbers/currency (5000 in "5000 ብር")  
   - `I-PRICE`: Currency unit ("ብር" in "5000 ብር")

3. **Location**:  
   - `B-LOC`: First word ("ቦሌ" in "ቦሌ አዲስ አበባ")  
   - `I-LOC`: Subsequent words

4. **O**: All non-entity words