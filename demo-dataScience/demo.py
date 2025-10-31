# streamlit run demo.py

import pdf_tools as pt
import streamlit as st

# --------------------------------------------------------------
# functions
# --------------------------------------------------------------
def reduce_pdf(pdf):
    scaled_doc, img_arr = pt.pdf_to_imgs(pdf)
    return pt.extract_relevant_pages(scaled_doc, img_arr)
    

# --------------------------------------------------------------
# page definition <- Start here!
# --------------------------------------------------------------
# TODO: figure out how to only run once (cache it!)
pt.init_tools()  
st.write('## PDF Reduction and Extraction Demo')
pdf_bytes = st.file_uploader('Upload PDF', type='pdf', accept_multiple_files=False)

# TODO: figure out how to deactivate the button while reduction is still running
if st.button('Reduce PDF'):
    if pdf_bytes == None:
        st.write('Add a PDF file first!')
        st.stop()
    else: 
        reduced_pdf = reduce_pdf(pdf_bytes)
        st.pdf(reduced_pdf, height=650)
        # TODO: use name of input PDF + _reduced.pdf
        st.download_button('Download reduced PDF', data=reduced_pdf, file_name='test.pdf')
