import re
question = "How has NVDA performed over the last 6 months?"
tickers = re.findall(r"\$([A-Z]{1,5})\b|\b([A-Z]{2,5})\b", question.upper())
print("Raw matches:", tickers)
tickers = [t[0] or t[1] for t in tickers]
print("Tickers after extraction:", tickers)
stopwords = {"HOW","HAS","THE","AND","FOR","OVER","LAST","VS","WHAT","IS","OF","IN","A","AN","TO","COMPARE","AM","YEAR","MONTH","MONTHS","YEARS","DAY","DAYS","ME","DO","MY","TODAY","MARKET","STOCK","PRICE","CURRENT","GET","CAN","YOU","TELL","ABOUT","SHOW","GIVE","HELP","IT","AT","BY","BE","ARE","WAS","ITS","ETF","CEO","CFO","USA","GDP","IMF","FED","SEC","IPO","YTD","EPS","PE"}
tickers = [t for t in tickers if t not in stopwords and len(t) >= 2]
print("Tickers after stopwords:", tickers)
