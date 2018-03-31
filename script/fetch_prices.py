import numpy as np
from datetime import datetime, timedelta
import pytz
import json
from urllib.request import Request, urlopen


headers = { "User-Agent": "Mozilla/5.0 (X11; Linux x86_64; rv:52.0) Gecko/20100101 Firefox/52.0" }

# "https://cex.io/api/ohlcv2/hd/20180329/BTC/USD",

def print_length(data):
	ohlcv = data["ohlcv"]
	print("1m: %s days" % (len(json.loads(ohlcv["data1m"]))/60/24))
	print("1h: %s days" % (len(json.loads(ohlcv["data1h"]))/24))
	print("1d: %s days" % len(json.loads(ohlcv["data1d"])))
	return data

def cast_entry(entry):
	return [int(entry[0])] + [float(v) for v in entry[1:]]

def cast_chart(chart):
	return [cast_entry(e) for e in chart]

def fetch_day(fc, tc, date=None):
	if date is None:
		url = "https://cex.io/api/ohlcv2/d/1m/%s/%s" % (fc, tc)
	else:
		ds = date.strftime('%Y%m%d')
		url = "https://cex.io/api/ohlcv2/hd/%s/%s/%s" % (ds, fc, tc)

	print("[ req ]: %s" % url)
	req = Request(url, headers=headers)
	data = json.loads(urlopen(req).read().decode("utf-8"))

	if date is None:
		chart = data["ohlcv"]
	else:
		chart = json.loads(data["ohlcv"]["data1m"])

	return cast_chart(chart)

rooms = [
	("BTC", "USD"),
	("BTC", "EUR"),
	("BTC", "GBP"),
	("BTC", "RUB"),
	("ETH", "USD"),
	("ETH", "EUR"),
	("ETH", "GBP"),
	("ETH", "BTC"),
	("BCH", "USD"),
	("BCH", "EUR"),
	("BCH", "GBP"),
	("BCH", "BTC"),
	("BTG", "USD"),
	("BTG", "EUR"),
	("BTG", "BTC"),
	("DASH", "USD"),
	("DASH", "EUR"),
	("DASH", "GBP"),
	("DASH", "BTC"),
	("XRP", "USD"),
	("XRP", "EUR"),
	("XRP", "BTC"),
	("XLM", "USD"),
	("XLM", "EUR"),
	("XLM", "BTC"),
	("ZEC", "USD"),
	("ZEC", "EUR"),
	("ZEC", "GBP"),
	("ZEC", "BTC"),
	("GHS", "BTC"),
]

def fetch_all(fc, tc, start_date=None):
	full_chart = []
	try:
		date = start_date
		while True:
			try:
				chart = fetch_day(fc, tc, date)
			except Exception as e:
				print(e)
				continue
			if chart is None or len(chart) == 0:
				print("[break] chart is empty")
				break
			full_chart.extend(reversed(chart))

			date = datetime.fromtimestamp(int(chart[0][0]))
			date = date.astimezone(pytz.utc)
			date -= timedelta(days=1)
	except Exception as e:
		print(e)
	finally:
		with open("%s-%s.json" % (fc, tc), "w") as f:
			f.write(json.dumps(list(reversed(full_chart))))

for fc, tc in rooms:
	fetch_all(fc, tc, datetime(year=2018, month=3, day=30))
