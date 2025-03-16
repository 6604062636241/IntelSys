import streamlit as st

st.title("svm info")

st.write("ผมเริ่มโดยการหา dataset ในเว็บ kaggle แล้วไปเจออันที่น่าสนใจคือ adult dataset ซึ่งเป็นข้อมูลเกี่ยวกับข้อมูลพื้นฐานของคนเช่น อายุ ชนิดการทำงาน ระดับการศึกษา สถานะสมรส อาชีพ")
st.image("./imgs/11.png", caption="", use_container_width=True)

st.write("ทำการโหลดโฟลเดอร์ adult.csv และเลือกเฉพาะคอลัมน์ที่ต้องการมี age workclass education marital.status occupation relationship race native.country income")
st.image("./imgs/12.png", caption="", use_container_width=True)

st.write("แปลงเป็นค่าตัวเลขโดยใช้ LabelEncoder เพื่อให้โมเดลใช้งานได้และค่าที่แปลงแล้วจะเก็บใน DataFrame และบันทึก LabelEncoder ในแต่ละคอลัมน์ไว้ใน encoder_dict เพื่อใช้ทีหลัง")
st.image("./imgs/13.png", caption="", use_container_width=True)

st.write("คอลัมน์ income เป็นคอลัมน์ที่ต้องทำนายว่าเงินเดือน <50K หรือ >50K ทำการแปลงเป็นตัวเลขด้วย LabelEncoder")
st.image("./imgs/14.png", caption="", use_container_width=True)

st.write("เอาค่าของคอลัมน์ age ทำการ Standardized ด้วย StandardScaler เพื่อให้มีค่าเป็น 0 และ SD เป็น 1 ช่วยเพิ่มประสิทธิภาพของโมเดล")
st.image("./imgs/15.png", caption="", use_container_width=True)

st.write("บันทึกโมเดล สเกลเลอร์ ตัวแปลงค่าของฟีเจอร์ ตัวแปลงของคอลัมน์ income ไว้ใช้ทีหลัง")
st.image("./imgs/16.png", caption="", use_container_width=True)



st.title("randomForest info")

st.write("ผมใช้ smote สำหรับสร้างข้อมูลจำลองในคลาสที่มีข้อมูลน้อย เพื่อแก้ปัญหาข้อมูลที่ไม่สมดุล")
st.image("./imgs/17.png", caption="", use_container_width=True)

st.write("ใช้ GridSearchCV เพื่อหาค่าพารามิเตอร์ที่ดีที่สุดสำหรับโมเดล โดยทดสอบค่าพารามิเตอร์ต่างๆ เช่น จำนวนต้นไม้ ความลึกสูงสุดของต้นไม้ จำนวนข้อมูลขั้นต่ำในการแบ่ง จำนวนข้อมูลขั้นต่ำในแต่ละใบไม้ และทำการเลือกค่าพารามิเตอร์ที่ให้ความแม่นยำสูงสุด")
st.image("./imgs/18.png", caption="", use_container_width=True)

st.write("save โมเดล สเกลเลอร์ ตัวแปลงค่าของฟีเจอร์ ตัวแปลงของคอลัมน์ income เพื่อให้นำไปใช้ทีหลังได้")
st.image("./imgs/19.png", caption="", use_container_width=True)
