import streamlit as st

st.title("CNN info")

st.write("ผมเริ่มโดยการหา dataset ในเว็บ kaggle แล้วไปเจออันที่น่าสนใจคือ Food-101 dataset ซึ่งเป็นข้อมูลเกี่ยวกับอาหาร 101 ชนิดที่มีภาพและ label โดยแต่ละชนิดมี 1,000 ภาพ แต่ละภาพมีองศามุมมองในการถ่ายของอาหารที่ไม่เหมือนกัน")
st.image("./imgs/1.png", caption="", use_container_width=True)

st.write("ทำการโหลดโฟลเดอร์ food101 และไปยังพาท images และลบภาพบางส่วนจากแต่ละหมวดหมู่ ลดที่คลาส apple_pie กับ churros อย่างละ 5%")
st.image("./imgs/2.png", caption="", use_container_width=True)

st.write("ทำ Data Augmentation เพื่อเพิ่มความยากในการฝึก")
st.image("./imgs/3.png", caption="", use_container_width=True)

st.write("ทำ Normalization โดยแปลงค่า pixel ของภาพจากช่วง 0-255 เป็น 0-1")
st.image("./imgs/4.png", caption="", use_container_width=True)

st.write("ปรับขนาดของภาพทั้งหมดเป็น 224x224 pixel และทำการ train")
st.image("./imgs/5.png", caption="", use_container_width=True)

st.write("ใช้ MobileNetV2 และตั้งให้ include_top=False เพื่อให้สามารถปรับและใช้กับคลาสที่เอามาได้ และตั้ง base_model.trainable=False เพื่อไม่ให้โมเดลฝึกในตอนแรก จะช่วยให้ฝึกได้เร็วขึ้น")
st.image("./imgs/6.png", caption="", use_container_width=True)

st.write("และฝึกโมเดลโดยใช้ GlobalAveragePooling ในการ pooling ข้อมูลจาก MobileNetV2 และทำการปรับ Dense ชั้น output ให้มีจำนวน units ตามจำนวนประเภทอาหาร และใช้ Dropout เพื่อป้องกัน overfitting")
st.image("./imgs/7.png", caption="", use_container_width=True)

st.write("ใช้ Adam optimizer ปรับ learning rate ให้น้อยเพื่อให้ตอนฝึกช่วยลดการข้ามค่าที่เหมาะสมใช้ Early Stopping เพื่อหยุดการฝึกหาก validation loss ไม่ลดลงหลังจาก 5 epoch ใช้ ReduceLROnPlateau ลด learning rate ถ้า validation loss ไม่ลดลงหลังจาก 3 epoch")
st.image("./imgs/8.png", caption="", use_container_width=True)

st.write("หลังจากฝึกในตอนแรกเสร็จแล้วจะทำการฝึก fine tuning โดยการเปิดการฝึกให้กับบาง layer ของ MobileNetV2")
st.image("./imgs/9.png", caption="", use_container_width=True)

st.write("เมื่อฝึก Fine-Tuning เสร็จแล้ว บันทึกโมเดลในไฟล์ food101_model.h5")
st.image("./imgs/10.png", caption="", use_container_width=True)
