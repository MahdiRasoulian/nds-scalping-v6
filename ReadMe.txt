cd "D:\Mahdi\New Backend\V-3\nds_bot"
.\venv\Scripts\activate
set FLASK_APP=main.py

dir /s /b *.py


python nds_mt5_bot.py --symbol XAUUSD! --timeframe H1 --history 500 --mode dry --risk 1.0 --sl_mult 2.


python nds_mt5_bot.py --symbol XAUUSD! --timeframe H1 --history 500 --mode live --risk 0.5 --sl_mult 2.5 --force

"C:/Program Files/MetaTrader 5/terminal64.exe"


python backtester.py
python test_compatibility.py
python main.py


pip install -r requirements.txt
pip install ta_lib-0.6.8-cp313-cp313-win_amd64.whl

python src/backtester/main.py

تمام کد را خز به خز تا آخر بخوان

تمام کد را ذخیره کن

بدون اینکه چیزی ناخواسته حذف شود تغییرات را اعمال کن و کد بهبهود یافته را ارایه کن:

