# Databricks notebook source
import pytz
from datetime import datetime
from functools import reduce
from pyspark.sql import functions as sf
from pyspark.sql import types as st
from pyspark.sql.window import Window

# COMMAND ----------

# db_connection = pymysql.connect(host='crawl.mysql.goibibo.com',
#                                 port=3306,
#                                 user='ranking',
#                                 passwd='sf4gR^T',
#                                 db='flw_augur')

# COMMAND ----------

today = str(datetime.now(pytz.timezone('Asia/Kolkata')).date())
dbutils.widgets.removeAll()
dbutils.widgets.text('1_vcid', '8717279093827200968')
dbutils.widgets.text('2_startDate', today)
dbutils.widgets.text('3_endDate', today)

# COMMAND ----------

vcid = dbutils.widgets.get('1_vcid')
startDate = datetime.strptime(dbutils.widgets.get('2_startDate'), '%Y-%m-%d').date()
endDate = datetime.strptime(dbutils.widgets.get('3_endDate'), '%Y-%m-%d').date()

print(vcid, startDate, endDate)

# COMMAND ----------

# DBTITLE 1,Searches
hotels_e_search = spark.table('hotels_e.search')

# COMMAND ----------

@sf.udf("string")
def get_sort(params):
  s = [param.value for param in params if param.key == 's']
  s = s[0].value if len(s) > 0 else None
  return s[0] if s is not None and len(s) > 0 else None

# COMMAND ----------

@sf.udf(st.ArrayType(st.StringType()))
def get_filters(filterInput):
  if filterInput is None:
    return []
  return [k for k,v in filterInput.asDict().items() if not(v is None or (type(v) == list and len(v) == 0))]

# COMMAND ----------

get_algo = sf.udf(lambda augurConfigId: augurConfigId[::-1].split('_')[0][::-1] if augurConfigId is not None else None)

# COMMAND ----------

@sf.udf(st.IntegerType())
def is_new_session(currentSearchTime, lastSearchTime, pid):
  #  pid is null or 0                                     inactivity of half hour
  if pid is None or pid == 0 or lastSearchTime is None or (currentSearchTime - lastSearchTime) > 30*60*1000:
    return 1
  return 0

# COMMAND ----------

@sf.udf(st.ArrayType(st.StringType()))
def flatten(hotels):
  return reduce(lambda x,y:x+y, hotels)

# COMMAND ----------

_searches = hotels_e_search \
                .filter((sf.col('day') >= startDate) & (sf.col('day') <= endDate)) \
                .filter(sf.col('vcid_key') == vcid) \
                .filter(sf.col('inputData.trackingId').isNotNull()) \
                .filter(sf.size(sf.col('outputData.rankList.ranks')) > 0) \
                .withColumn('sort', get_sort(sf.col('inputData.params'))) \
                .withColumn('filters', get_filters(sf.col('inputData.filterInput'))) \
                .withColumn('algo', get_algo(sf.col('outputData.augurConfigID'))) \
                .select(sf.col('day').alias('searchDay'),
                        sf.from_utc_timestamp(sf.col('eventTime').cast(st.TimestampType()), 'IST').alias('searchTime'),
                        sf.col('currentTime').alias('currentSearchTime'),
                        sf.col('vcid_key').alias('vcid'),
                        sf.col('inputData.flavour').alias('flavour'),
                        sf.col('inputData.checkIn').alias('checkIn'),
                        sf.col('inputData.checkOut').alias('checkOut'),
                        sf.col('inputData.pax').alias('pax'),
                        sf.col('inputData.userId').alias('userId'),
                        sf.col('inputData.email').alias('email'),
                        sf.col('inputData.trackingId').alias('trackingId'),
                        sf.col('sort'),
                        sf.col('filters'),
                        sf.col('inputData.filterInput').alias('filterInput'),
                        sf.col('algo'),
                        sf.col('pid'),
                        sf.col('outputData.rankList.ranks').alias('ranks'))

# display(_searches)

# COMMAND ----------

columnsToPartitionOn = ['vcid', 'flavour', 'checkIn', 'checkOut', 'pax', 'userId', 'email', 'trackingId', 'sort', 'filters', 'filterInput',
                        'algo']
searches = _searches.withColumn('lastSearchTime', sf.lag('currentSearchTime') \
                                                    .over(Window.partitionBy(columnsToPartitionOn) \
                                                    .orderBy('currentSearchTime'))) \
                   .withColumn('isNewSession', is_new_session(sf.col('currentSearchTime'), sf.col('lastSearchTime'), sf.col('pid'))) \
                   .withColumn('sessionId', sf.sum('isNewSession') \
                                              .over(Window.partitionBy(columnsToPartitionOn) \
                                              .orderBy('currentSearchTime'))) \
                   .groupBy(columnsToPartitionOn + ['sessionId']) \
                   .agg(sf.min('searchTime').alias('searchTime'),
                        sf.collect_list('ranks').alias('hotels')) \
                   .withColumn('hotels', flatten(sf.col('hotels')))

display(searches)

# COMMAND ----------

# DBTITLE 1,Details
hotels_e_detail = spark.table('hotels_e.detail')
# display(hotels_e_detail)

# COMMAND ----------

details = hotels_e_detail.filter((sf.col('day') >= startDate) & (sf.col('day') <= endDate)) \
                         .filter(sf.col('detailsDataLog.cityId').cast(st.StringType()) == vcid) \
                         .filter(sf.col('detailsDataLog.trackingID').isNotNull()) \
                         .select(sf.from_utc_timestamp(sf.col('eventTime').cast(st.TimestampType()), 'IST').alias('detailTime'),
                                 sf.col('detailsDataLog.cityId').cast(st.StringType()).alias('vcid'),
                                 sf.col('detailsDataLog.flavour').alias('flavour'),
                                 sf.col('detailsDataLog.checkIn').alias('checkIn'),
                                 sf.col('detailsDataLog.checkOut').alias('checkOut'),
                                 sf.col('detailsDataLog.pax').alias('pax'),
                                 sf.col('detailsDataLog.userId').alias('userId'),
                                 sf.col('detailsDataLog.email').alias('email'),
                                 sf.col('detailsDataLog.trackingID').alias('trackingId'),
                                 sf.col('detailsDataLog.hotelId').cast(st.StringType()).alias('vhid'))

display(details)

# COMMAND ----------

# DBTITLE 1,Bookings
hotels_e_etl = spark.table('hotels_e.etl')
display(hotels_e_etl)

# COMMAND ----------

def get_supply_margin(vendor, paymode, grand_total, netamt, bookingcharges, totaltaxcharges_pah):
  if vendor == 'bkg':
    return (grand_total * 6.36) / 100
  if vendor == 'ihg':
    return (grand_total * 8.8) / 100
  if vendor == 'exp':
    return ((grand_total - (netamt + bookingcharges)) * 0.65) / 1.18
  else:
    _netamt = (netamt + (totaltaxcharges_pah or 0)) if vendor is not None and vendor.lower() == 'ingoibibo' and paymode in (1,5) else netamt
    return (grand_total - (_netamt + bookingcharges)) / 1.18

# COMMAND ----------

def get_final_amt(paymode, netamt, totaltaxcharges_pah):
  return netamt + (totaltaxcharges_pah or 0) if paymode in (1,5) else netamt

# COMMAND ----------

def get_final_net_amt(vendor, paymode, voyagerhotelid, vendor_amount, netamt, bookingcharges, totaltaxcharges_pah, totaltaxcharges):
  commission = totaltaxcharges - bookingcharges
  trust_vhids = ['464241129666544', '738632572558523', '613301088464001', '377277701863009', '903200440281944', '855892936863786',
                 '710317332837447', '686071580782092', '547280575134534', '674803750424735', '826089272943386', '118081405006790',
                 '282760921508358', '824000701458379', '160517388704968', '647932802151791', '726440044168123', '620832173637029',
                 '257244117119279', '156655907264922', '280359415060074', '549769109205954']
  if vendor == 'bkg':
    return (vendor_amount - commission) * 0.95 + commission
  if vendor == 'ihg':
    return (vendor_amount - commission) * 0.90 + commission
  if vendor == 'tbo':
    return (vendor_amount - commission) * 0.90 + commission
  if vendor == 'exp':
    return (vendor_amount - commission) * 0.91 + commission
  if vendor == 'trust':
    if voyagerhotelid[:15] in trust_vhids:
      return (vendor_amount - commission) * 0.85 + commission
    else:
      return (vendor_amount - commission) * 0.90 + commission
  else:
    return get_final_amt(paymode, netamt, totaltaxcharges_pah)

# COMMAND ----------

get_confirmed_bookings = sf.udf(lambda status,room_nights: room_nights if status in ('to deliver','manual') or status.endswith('reserved') else 0, 'int')

# COMMAND ----------

get_cancelled_bookings = sf.udf(lambda status, room_nights: room_nights if status.contains('cancel') or status.endswith('refund') else 0, 'int')

# COMMAND ----------

def get_gmv(hotel_country, confirmed_bookings, vendor_amount, bookingcharges):
  if hotel_country.lower() == 'india' and confirmed_bookings > 0:
    return vendor_amount + bookingcharges
  if hotel_country.lower() != 'india' and confirmed_bookings > 0:
    return grand_total - bookingcharges
  return 0

case when lower(hotel_country) = 'india' and etl.is_akb = 0 and confirmed_bookings > 0 then (vendor_amount - final_net_amt)/1.18
           when lower(hotel_country) != 'india' and confirmed_bookings > 0 then supply_margin
      else 0 end as margin,


# COMMAND ----------

bookings = hotels_e_etl.filter((sf.col('month') >= int(datetime.strftime(startDate, '%Y%m'))) & 
                               (sf.col('month') <= int(datetime.strftime(endDate, '%Y%m')))) \
                       .filter((sf.col('day') >= startDate) & (sf.col('day') <= endDate)) \
                       .filter(sf.col('voyagercityid').cast(st.StringType()) == vcid) \
                       .filter(sf.col('tid').isNotNull()) \
                       .select(sf.col('bookingdate').alias('bookingTime'),
                               sf.col('voyagercityid').cast(st.StringType()).alias('vcid'),
                               sf.col('voyagerhotelid').cast(st.StringType()).alias('vhid'),
                               sf.col('flavour'),
                               sf.col('checkin'),
                               sf.col('checkout'),
                               sf.col('rooms'),
                               sf.col('adults'),
                               sf.col('children'),
                               sf.col('uid').alias('userId'),
                               sf.col('email'),
                               sf.col('tid').alias('trackingId'),
                               sf.col('status'),
                               sf.col('paymode'))

display(bookings)

# COMMAND ----------

# DBTITLE 1,Rough
int(datetime.strftime(startDate, '%Y%m'))

# COMMAND ----------

filterInput = t[0].inputData.filterInput
filterInput.schema
# for f in filterInput:
#   print(f)

# COMMAND ----------

[k for k,v in filterInput.asDict().items() if not(v is None or (type(v) == list and len(v) == 0))]

# COMMAND ----------

_s = "dom_base_upr_default"
_s[::-1].split('_')[0][::-1]

# COMMAND ----------

sf.split(sf.reverse("dom_base_upr_default"), "_")[0]
