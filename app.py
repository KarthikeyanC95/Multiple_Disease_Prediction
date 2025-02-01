import os
import pickle
import streamlit as st
from streamlit_lottie import st_lottie
from PIL import Image
from streamlit_option_menu import option_menu
parkinsons_model = pickle.load(open("E:\Multiple disease prediction\parkinson_model.sav", "rb"))
liver_model = pickle.load(open("E:\Multiple disease prediction\liver_model.sav", "rb"))
Kidney_model = pickle.load(open("E:\Multiple disease prediction\kidney_model.sav", "rb"))

with st.sidebar:
    selected = option_menu('Multiple Disease Prediction System',
                           
                           ['Kidney Prediction',
                            'Liver Prediction',
                            'Parkinsons Prediction'],
                            menu_icon='hosiptial-fill',
                            icons= ['activity', 'heart', 'person'],
                            default_index = 0)
#Kidney Prediction
if selected == 'Kidney Prediction':
    
    st.title(":red[Kidney Disease Prediction using ML]")
    image = Image.open('Kidney.jpg')
    st.image(image, caption='Kidney disease')
    name = st.text_input("Name:")

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        age = st.number_input('Age')

    with col2:
        bp = st.number_input('Blood Pressure')

    with col3:
        sg = st.number_input('Specific Gravity')

    with col4:
        al = st.number_input('Albumin')

    with col5:
        su = st.number_input('Sugar')

    with col1:
        rbc = st.selectbox("Red Blood Cells", ["Normal", "Abnormal", "Null"])
        rbc = 1 if rbc == "Normal" else 0

    with col2:
        pc = st.selectbox("Pus Cells", ["Normal", "Abnormal", "Null"])
        pc = 1 if pc == "Normal" else 0

    with col3:
        pcc = st.selectbox("Pus Cell Clumps", ["Present", "Not Present", "Null"])
        pcc = 1 if pcc == "Present" else 0

    with col4:
        ba = st.selectbox("Bacteria", ["Present", "Not Present", "Null"])
        ba = 1 if ba == "Present" else 0

    with col5:
        bgr = st.number_input('Blood Glucose Random')

    with col1:
        bu = st.number_input('Blood Urea')

    with col2:
        sc = st.number_input('Serum Creatinine')

    with col3:
        sod = st.number_input('Sodium')

    with col4:
        pot = st.number_input('Potassium')

    with col5:
        hemo = st.number_input('Haemoglobin')

    with col1:
        pcv = st.number_input('Packet Cell Volume')

    with col2:
        wc = st.number_input('White Blood Cell Count')

    with col3:
        rc = st.number_input('Red Blood Cell Count')

    with col4:
        htn = st.selectbox("Hypertension", ["Yes", "No", "Null"])
        htn = 1 if htn == "Yes" else 0

    with col5:
        dm = st.selectbox("Diabetes Mellitus", ["Yes", "No", "Null"])
        dm = 1 if dm == "Yes" else 0

    with col1:
        cad = st.selectbox("Coronary Artery Disease", ["Yes", "No", "Null"])
        cad = 1 if cad == "Yes" else 0

    with col2:
        appet = st.selectbox("Appetite", ["Good", "Poor", "Null"])
        appet = 1 if appet == "Good" else 0

    with col3:
        pe = st.selectbox("Pedal Edema", ["Yes", "No", "Null"])
        pe = 1 if pe == "Yes" else 0
    with col4:
        ane = st.selectbox("Anemia", ["Yes", "No", "Null"])
        ane = 1 if ane == "Yes" else 0

    # code for Prediction
    kindey_diagnosis = ''

    # creating a button for Prediction    
    if st.button("Kidney's Test Result"):

        user_input = [age, bp, sg, al, su, rbc, pc, pcc, ba, bgr, bu, sc, sod, pot, hemo, pcv, wc, rc, htn, dm, cad, appet, pe, ane]

        prediction = Kidney_model.predict([user_input])

        if prediction[0] == 1:
            kindey_diagnosis = "The person has Kidney's disease"
            image = Image.open('positive.jpg')
            st.image(image, caption='')
        else:
            kindey_diagnosis = "The person does not have Kidney's disease"
            image = Image.open('negative.jpg')
            st.image(image, caption='')
    st.success('Hi'+' '+name+' , ' + kindey_diagnosis)

##Liver Prediction
if selected == 'Liver Prediction': 
    st.title(":red[Liver disease prediction using ML]")
    image = Image.open('liver.jpg')
    st.image(image, caption='Liver disease prediction.')
    
    name = st.text_input("Name:")
    col1, col2, col3 = st.columns(3)

    with col1:
        Gender_Male = 0
        display = ("male", "female")
        options = list(range(len(display)))
        value = st.selectbox("Gender", options, format_func=lambda x: display[x])
        if value == "male":
            Gender_Male = 0
        elif value == "female":
            Gender_Male = 1
    with col2:
        age = st.number_input("Age") # 2 
    with col3:
        Total_Bilirubin = st.number_input("Total Bilirubin") # 3
    with col1:
        Direct_Bilirubin = st.number_input("Direct Bilirubin")# 4

    with col2:
        Alkaline_Phosphotase = st.number_input("Alkaline Phosphotase") # 5
    with col3:
        Alamine_Aminotransferase = st.number_input("Alamine Aminotransferase") # 6
    with col1:
        Aspartate_Aminotransferase = st.number_input("Aspartate Aminotransferase") # 7
    with col2:
        Total_Protiens = st.number_input("Total Protiens")# 8
    with col3:
        Albumin = st.number_input("Albumin") # 9
    with col1:
        Albumin_and_Globulin_Ratio = st.number_input("Albumin and Globulin Ratio") # 10 
    # code for prediction
    liver_dig = ''

    # button
    if st.button("Liver test result"):
        user_input = [Gender_Male,age,Total_Bilirubin,Direct_Bilirubin,Alkaline_Phosphotase,Alamine_Aminotransferase,Aspartate_Aminotransferase,Total_Protiens,Albumin,Albumin_and_Globulin_Ratio]
        
        liver_prediction = liver_model.predict([user_input])

        # after the prediction is done if the value in the list at index is 0 is 1 then the person is diabetic
        if liver_prediction[0] == 1:
            image = Image.open('positive.jpg')
            st.image(image, caption='')
            liver_dig = "we are really sorry to say but it seems like you have liver disease."
        else:
            image = Image.open('negative.jpg')
            st.image(image, caption='')
            liver_dig = "Congratulation , You don't have liver disease."
        st.success('Hi'+' '+name+' , ' + liver_dig)


#parkinsions Prediction
if selected == "Parkinsons Prediction":

    #title
    st.title(":red[Parkinson's Disease Prediction using ML]")
    image = Image.open('p1.jpg')
    st.image(image, caption='parkinsons disease')
    name = st.text_input("Name:")
    col1, col2, col3 = st.columns(3)
    with col1:
        Fo = st.number_input("MDVP:Fo(Hz)")
    with col2:
        Fhi = st.number_input("MDVP:Fhi(Hz)")
    with col3:
        Flo = st.number_input("MDVP:Flo(Hz)")
    with col1:
        Jitter_percent = st.number_input("MDVP:Jitter(%)")
    with col2:
        Jitter_Abs = st.number_input("MDVP:Jitter(Abs)")
    with col3:
        RAP = st.number_input("MDVP:RAP")
    

    with col2:
        PPQ = st.number_input("MDVP:PPQ ")
    with col3:
        DDP = st.number_input("Jitter:DDP")
    with col1:
        Shimmer = st.number_input("MDVP:Shimmer")
    with col2:
        Shimmer_dB = st.number_input("MDVP:Shimmer(dB)")
    with col3:
        APQ3 = st.number_input("Shimmer:APQ3")
    with col1:
        APQ5 = st.number_input("Shimmer:APQ5")
    with col2:
        APQ = st.number_input("MDVP:APQ")
    with col3:
        DDA = st.number_input("Shimmer:DDA")
    with col1:
        NHR = st.number_input("NHR")
    with col2:
        HNR = st.number_input("HNR")

    
    with col2:
        RPDE = st.number_input("RPDE")
    with col3:
        DFA = st.number_input("DFA")
    with col1:
        spread1 = st.number_input("spread1")
    with col1:
        spread2 = st.number_input("spread2")
    with col3:
        D2 = st.number_input("D2")
    with col1:
        PPE = st.number_input("PPE")

    # code for prediction
    parkinson_dig = ''

    # creating buttom for the prediction
    if st.button("Parkinsons test result"):

        user_input=[Fo, Fhi, Flo, Jitter_percent, Jitter_Abs, 
                    RAP, PPQ, DDP, Shimmer, Shimmer_dB, 
                    APQ3, APQ5, APQ, DDA, NHR, 
                    HNR, RPDE, DFA, spread1, spread2, 
                    D2, PPE]
        
        parkinson_prediction = parkinsons_model.predict([user_input])

        if parkinson_prediction[0] == 1:
            parkinson_dig = 'we are really sorry to say but it seems like you have Parkinson disease'
            image = Image.open('positive.jpg')
            st.image(image, caption='')
        else:
            parkinson_dig = "Congratulation , You don't have Parkinson disease"
            image = Image.open('negative.jpg')
            st.image(image, caption='')
        st.success('Hi'+' '+name+' , ' + parkinson_dig)


