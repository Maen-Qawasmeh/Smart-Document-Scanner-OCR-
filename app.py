
import json
import hashlib

import cv2
import numpy as np
import pandas as pd
import streamlit as st

from src.pipeline import DocumentPipeline


# ============================================================
# 1. Page Configuration
# ============================================================

st.set_page_config(
    page_title="Smart Document Scanner & OCR",
    page_icon="📄",
    layout="wide"
)


# ============================================================
# 2. Session State Initialization
# ============================================================

if "ocr_result" not in st.session_state:
    st.session_state.ocr_result = None

if "uploaded_file_id" not in st.session_state:
    st.session_state.uploaded_file_id = None

if "last_settings" not in st.session_state:
    st.session_state.last_settings = None


# ============================================================
# 3. Load OCR Pipeline
# ============================================================

@st.cache_resource
def load_pipeline():
    """
    Initialize and cache the OCR pipeline.
    """
    return DocumentPipeline(
        language="en",
        device="cpu"
    )


# ============================================================
# 4. Application Header
# ============================================================

st.title("📄 Smart Document Scanner & OCR")

st.write(
    "Upload a document, correct its perspective, "
    "enhance image quality, and extract text using PaddleOCR."
)


# ============================================================
# 5. Sidebar Settings
# ============================================================

st.sidebar.header("Processing Settings")

auto_scan = st.sidebar.checkbox(
    "Enable Document Scanning",
    value=False,
    help="Detect document corners and correct perspective."
)

preprocessing = st.sidebar.selectbox(
    "Image Preprocessing",
    options=[
        "original",
        "grayscale",
        "clahe",
        "adaptive"
    ]
)

current_settings = {
    "auto_scan": auto_scan,
    "preprocessing": preprocessing
}


# ============================================================
# 6. Document Upload
# ============================================================

uploaded_file = st.file_uploader(
    "Upload Document Image",
    type=["jpg", "jpeg", "png"]
)


if uploaded_file is not None:

    # Read uploaded image bytes
    file_content = uploaded_file.getvalue()

    # Create a unique identifier for the uploaded image
    current_file_id = hashlib.sha256(
        file_content
    ).hexdigest()

    # Clear old results when a different image is uploaded
    if current_file_id != st.session_state.uploaded_file_id:
        st.session_state.uploaded_file_id = current_file_id
        st.session_state.ocr_result = None
        st.session_state.last_settings = None

    # Decode image
    file_bytes = np.frombuffer(
        file_content,
        dtype=np.uint8
    )

    image = cv2.imdecode(
        file_bytes,
        cv2.IMREAD_COLOR
    )

    if image is None:
        st.error("Unable to read the uploaded image.")
        st.stop()

    # Display original document
    st.subheader("Original Document")

    st.image(
        cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
        width=500
    )


    # ========================================================
    # 7. Document Processing
    # ========================================================

    if st.button("Scan & Extract Text", type="primary"):

        try:
            with st.spinner("Processing document..."):

                pipeline = load_pipeline()

                result = pipeline.process(
                    image,
                    auto_scan=auto_scan,
                    preprocessing=preprocessing
                )

                # Save results in session state
                st.session_state.ocr_result = result

                # Save settings used for processing
                st.session_state.last_settings = (
                    current_settings.copy()
                )

        except Exception as error:
            st.session_state.ocr_result = None
            st.session_state.last_settings = None

            st.error(
                f"Document processing failed: {error}"
            )


    # ========================================================
    # 8. Display Saved Results
    # ========================================================

    if st.session_state.ocr_result is not None:

        result = st.session_state.ocr_result

        st.success("Document processed successfully!")

        # Warn when current settings differ from saved settings
        if current_settings != st.session_state.last_settings:
            st.warning(
                "Processing settings have changed. "
                "The displayed results belong to the previous "
                "settings. Click 'Scan & Extract Text' to update."
            )

        st.write(
            "Scan Status:",
            result["scan_status"]
        )

        st.write(
            "Preprocessing:",
            result["preprocessing"]
        )


        # ====================================================
        # 9. Display Scanned and Processed Images
        # ====================================================

        col1, col2 = st.columns(2)

        with col1:

            st.subheader("Scanned Document")

            st.image(
                cv2.cvtColor(
                    result["scanned_image"],
                    cv2.COLOR_BGR2RGB
                ),
                width=400
            )

        with col2:

            st.subheader("Processed Document")

            processed = result["processed_image"]

            if processed.ndim == 2:

                st.image(
                    processed,
                    clamp=True,
                    width=400
                )

            else:

                st.image(
                    cv2.cvtColor(
                        processed,
                        cv2.COLOR_BGR2RGB
                    ),
                    width=400
                )


        # ====================================================
        # 10. Download Scanned Document
        # ====================================================

        success, encoded_image = cv2.imencode(
            ".png",
            result["scanned_image"]
        )

        if success:

            st.download_button(
                label="Download Scanned Image",
                data=encoded_image.tobytes(),
                file_name="scanned_document.png",
                mime="image/png"
            )

        else:

            st.error("Unable to encode scanned image.")


        # ====================================================
        # 11. Extracted Text
        # ====================================================

        text_results = result["text_results"]

        extracted_text = "\n".join(
            item["text"] for item in text_results
        )

        st.subheader("Extracted Text")

        st.text_area(
            "OCR Output",
            value=extracted_text,
            height=300
        )

        st.metric(
            "Recognized Text Segments",
            len(text_results)
        )


        # ====================================================
        # 12. OCR Confidence Analysis
        # ====================================================

        st.subheader("OCR Confidence Analysis")

        if text_results:

            confidence_df = pd.DataFrame(text_results)

            confidence_df["Confidence (%)"] = (
                confidence_df["confidence"] * 100
            ).round(2)

            confidence_df = confidence_df.rename(
                columns={
                    "text": "Recognized Text"
                }
            )

            st.dataframe(
                confidence_df[
                    ["Recognized Text", "Confidence (%)"]
                ],
                use_container_width=True,
                hide_index=True
            )

            # Calculate confidence metrics
            avg_confidence = (
                confidence_df["confidence"].mean() * 100
            )

            min_confidence = (
                confidence_df["confidence"].min() * 100
            )

            metric_col1, metric_col2 = st.columns(2)

            with metric_col1:

                st.metric(
                    "Average Confidence",
                    f"{avg_confidence:.2f}%"
                )

            with metric_col2:

                st.metric(
                    "Minimum Confidence",
                    f"{min_confidence:.2f}%"
                )

        else:

            st.info(
                "No text was detected in this document."
            )


        # ====================================================
        # 13. Prepare JSON Export
        # ====================================================

        json_data = {
            "scan_status": result["scan_status"],
            "preprocessing": result["preprocessing"],
            "total_text_segments": len(text_results),
            "ocr_results": text_results
        }


        # ====================================================
        # 14. Download OCR Results
        # ====================================================

        st.subheader("Download OCR Results")

        col_txt, col_json = st.columns(2)

        with col_txt:

            st.download_button(
                label="Download TXT",
                data=extracted_text.encode("utf-8"),
                file_name="extracted_text.txt",
                mime="text/plain"
            )

        with col_json:

            st.download_button(
                label="Download JSON",
                data=json.dumps(
                    json_data,
                    indent=4,
                    ensure_ascii=False
                ).encode("utf-8"),
                file_name="ocr_results.json",
                mime="application/json"
            )

else:
    # Clear results when no document is uploaded
    st.session_state.uploaded_file_id = None
    st.session_state.ocr_result = None
    st.session_state.last_settings = None
