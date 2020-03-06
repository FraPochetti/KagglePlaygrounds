import sys
from awsglue.transforms import *
from awsglue.utils import getResolvedOptions
from pyspark.context import SparkContext
from awsglue.context import GlueContext
from awsglue.job import Job
from pyspark.sql.functions import col, when, to_date
from awsglue.dynamicframe import DynamicFrame

def flatten_df(nested_df):
    flat_cols = [c[0] for c in nested_df.dtypes if c[1][:6] != 'struct']
    nested_cols = [c[0] for c in nested_df.dtypes if c[1][:6] == 'struct']

    flat_df = nested_df.select(flat_cols +
                               [col(nc+'.'+c).alias(nc+'_'+c)
                                for nc in nested_cols
                                for c in nested_df.select(nc+'.*').columns])
    return flat_df

args = getResolvedOptions(sys.argv, ['JOB_NAME'])

sc = SparkContext()
glueContext = GlueContext(sc)
spark = glueContext.spark_session
job = Job(glueContext)
job.init(args['JOB_NAME'], args)

ds0 = glueContext.create_dynamic_frame.from_catalog(database = "covid19", 
                                                    table_name = "pochetti_covid_19_input", 
                                                    transformation_ctx = "ds0")
ds0 = ds0.select_fields(['Province-State', 'Country-Region', 'Lat', 'Long', 'Date', 'Confirmed', 'Deaths', 'Recovered', 'id'])

df = ds0.toDF()
df = flatten_df(df)

df = df.select(
        when(df['Province-State'].isin({'NULL', '', 'missing', '--'}), None)
        .otherwise(df['Province-State']).alias('Province-State'), 
        
        'Country-Region',
        'Lat',
        'Long',
        
        when(df['Recovered_int'].isNull(), 0).otherwise(df['Recovered_int']).alias('Recovered'),
        when(df['Confirmed_int'].isNull(), 0).otherwise(df['Confirmed_int']).alias('Confirmed'),
        when(df['Deaths_int'].isNull(), 0).otherwise(df['Deaths_int']).alias('Deaths'),
        
        when(to_date(col("Date"),"yyyy-MM-dd").isNotNull(), 
             to_date(col("Date"),"yyyy-MM-dd"))
        .when(to_date(col("Date"),"yyyy/MM/dd").isNotNull(),
              to_date(col("Date"),"yyyy/MM/dd"))
        .when(to_date(col("Date"),"yyyy-MMM-dd").isNotNull(),
              to_date(col("Date"),"yyyy-MMM-dd"))    
        .when(to_date(col("Date"),"yyyy/MMMM/dd").isNotNull(),
              to_date(col("Date"),"yyyy/MMMM/dd"))    
        .when(to_date(col("Date"),"yyyy, MMMM, dd").isNotNull(),
              to_date(col("Date"),"yyyy, MMMM, dd"))
        .otherwise("Unknown Format").alias("Date"),
        
        'id'
)

datasource_transformed = DynamicFrame.fromDF(df, glueContext, "ds0")

datasink2 = glueContext.write_dynamic_frame.from_options(frame = datasource_transformed, connection_type = "s3", 
                                                        connection_options = {"path": "s3://pochetti-covid-19-output"}, 
                                                        format = "json", transformation_ctx = "datasink2")

job.commit()