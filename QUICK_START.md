# Quick Start Guide - DR-TB Prediction System

## 🚀 Starting the Application

### Step 1: Activate Virtual Environment

Open your terminal and navigate to the project directory:

```bash
cd "/home/santhosh/Desktop/DR-TB research project"
```

Activate the virtual environment:

```bash
source venv/bin/activate
```

You should see `(venv)` in your terminal prompt.

### Step 2: Start the Web Application

**Option A: Using the startup script (Easiest)**
```bash
./run_app.sh
```

**Option B: Manual start**
```bash
streamlit run app.py
```

### Step 3: Access the Application

The app will automatically open in your browser at:
- **URL**: `http://localhost:8501`

If it doesn't open automatically, copy the URL from the terminal and paste it in your browser.

---

## 🧪 Testing the Application

### Test 1: Basic Functionality Check

1. **Upload a Test Image**
   - Use any chest X-ray image (PNG, JPG, or JPEG)
   - You can use sample images from: `TB_Chest_Radiography_Database/Tuberculosis/` or `TB_Chest_Radiography_Database/Normal/`

2. **Fill in Clinical Data**
   - Age: 45
   - Gender: Male
   - Region: Asia
   - Leave other checkboxes unchecked for a basic test

3. **Genomic Mutations (Optional)**
   - Leave all unchecked for initial test

4. **Run Prediction**
   - Click "🔬 Run Prediction" button
   - Wait a few seconds for processing

5. **Check Results**
   - You should see a prediction (DR-TB or Normal)
   - Probability and confidence scores
   - Risk level indicator

### Test 2: Detailed Report

1. After getting a prediction, check "Show Detailed Report"
2. Review all sections:
   - Prediction Summary
   - Risk Factors
   - Genomic Analysis
   - Modality Contributions
   - Clinical Recommendations

### Test 3: With Genomic Data

1. Upload an image
2. Fill clinical data
3. Check some genomic mutations (e.g., rpoB S531L, katG S315T)
4. Run prediction
5. Verify that mutations appear in the detailed report

### Test 4: Export Report

1. After getting a prediction with detailed report
2. Click "📥 Download Report (TXT)"
3. Verify the file downloads correctly

---

## ✅ Expected Behavior

### When Everything Works:

- ✅ Model loads successfully (check sidebar for model info)
- ✅ Image uploads and displays correctly
- ✅ Prediction completes in 2-5 seconds
- ✅ Results show with proper formatting
- ✅ Detailed report expands correctly
- ✅ No error messages in terminal or browser

### Common Issues:

**Issue**: "Model could not be loaded"
- **Solution**: Ensure model files (`.pth`) exist in `results/models/` directory

**Issue**: "CUDA out of memory"
- **Solution**: The app will automatically fall back to CPU (slower but works)

**Issue**: Image upload fails
- **Solution**: Check file format (PNG, JPG, JPEG) and size (<10MB)

---

## 📝 Sample Test Data

### Test Case 1: Normal Patient
- Age: 30
- Gender: Female
- Region: Asia
- No medical history
- No mutations
- **Expected**: Normal prediction with low risk

### Test Case 2: High-Risk DR-TB
- Age: 55
- Gender: Male
- Region: Africa
- Previous TB Treatment: ✓
- HIV Positive: ✓
- rpoB S531L: ✓
- katG S315T: ✓
- **Expected**: DR-TB prediction with high risk

### Test Case 3: Moderate Risk
- Age: 45
- Gender: Male
- Region: Asia
- Previous TB Treatment: ✓
- No mutations
- **Expected**: Moderate risk prediction

---

## 🔍 Verification Checklist

Before considering the app ready:

- [ ] Virtual environment activates without errors
- [ ] Streamlit app starts successfully
- [ ] Browser opens automatically
- [ ] Model loads (check sidebar)
- [ ] Can upload an image
- [ ] Can fill clinical form
- [ ] Can select genomic mutations
- [ ] Prediction runs without errors
- [ ] Results display correctly
- [ ] Detailed report shows all sections
- [ ] Report can be downloaded

---

## 🛑 Stopping the Application

To stop the Streamlit app:
1. Go to the terminal where it's running
2. Press `Ctrl + C`
3. Type `deactivate` to exit the virtual environment (optional)

---

## 📞 Need Help?

If you encounter issues:
1. Check the terminal for error messages
2. Verify model files exist in `results/models/`
3. Ensure virtual environment is activated
4. Check that all dependencies are installed: `pip list`

