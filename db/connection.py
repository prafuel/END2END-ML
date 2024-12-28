import mysql.connector

class connector:
	def __init__(self):
		try:
			self.conn = mysql.connector.connect(
				host="localhost",
				user="prafull",
				password="", 
				database="creditcard"
			)

			self.cursor = self.conn.cursor()
		except:
			print("Connection Failed...")
		else:
			print("Connection Successful...")

