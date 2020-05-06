%windir%\System32\cmd.exe "/K" D:\Users\shadi\Anaconda3\Scripts\activate.bat D:\Users\shadi\Anaconda3
call activate D:\Users\shadi\.conda\pycharmenv
streamlit run app.py
call conda deactivate