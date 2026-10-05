TypeError: This app has encountered an error. The original error message is redacted to prevent data leaks. Full error details have been recorded in the logs (if you're on Streamlit Cloud, click on 'Manage app' in the lower right of your app).
Traceback:
File "/mount/src/spam-fake-detection/rontend/app.py", line 20, in <module>
    st.markdown("""
    ~~~~~~~~~~~^^^^
    <style>
    ^^^^^^^
    ...<25 lines>...
    </style>
    ^^^^^^^^
    """, unsafe_unsafe_html=True)
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
