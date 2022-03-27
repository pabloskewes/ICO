import pandas as pd
import os

ROOT_DIR = os.path.abspath('../')
DATA_DIR = os.path.join(ROOT_DIR, 'data')

CUSTOMER_DIR = os.path.join(DATA_DIR, '2_detail_table_customers.xls')
VEHICLES_DIR = os.path.join(DATA_DIR, '3_detail_table_vehicles.xls')
DEPOTS_DIR = os.path.join(DATA_DIR, '4_detail_table_depots.xls')
CONSTRAINTS_DIR = os.path.join(DATA_DIR, '5_detail_table_constraints_sdvrp.xls')
DEPOTS_DISTANCES_DIR = os.path.join(DATA_DIR, '6_detail_table_cust_depots_distances.xls')
CUSTOMER_DISTANCES_DIR = os.path.join(DATA_DIR, '7_detail_table_cust_cust_distances.xls')


customers = pd.read_excel(CUSTOMER_DIR)
vehicles = pd.read_excel(VEHICLES_DIR)
depots = pd.read_excel(DEPOTS_DIR)
constraints = pd.read_excel(CONSTRAINTS_DIR)
depots_dist = pd.read_excel(DEPOTS_DISTANCES_DIR)
customers_dist = pd.read_excel(CUSTOMER_DISTANCES_DIR)

# process customers data
customers.drop_duplicates(['CUSTOMER_CODE'],inplace=True)
customers.drop(['CUSTOMER_LATITUDE','CUSTOMER_LONGITUDE','NUMBER_OF_ARTICLES'],axis=1,inplace=True)
# process vehicle data
vehicles.drop(['ROUTE_ID','RESULT_VEHICLE_TOTAL_DRIVING_TIME_MIN','RESULT_VEHICLE_TOTAL_DELIVERY_TIME_MIN','RESULT_VEHICLE_TOTAL_ACTIVE_TIME_MIN','RESULT_VEHICLE_DRIVING_WEIGHT_KG','RESULT_VEHICLE_DRIVING_VOLUME_M3','RESULT_VEHICLE_FINAL_COST_KM'],axis=1,inplace=True)
vehicles.drop_duplicates(['VEHICLE_CODE'],inplace=True)

# combine the depots_dist and the customers_dist
depots_dist.rename(columns={'DEPOT_CODE':'CUSTOMER_CODE_FROM','CUSTOMER_CODE':'CUSTOMER_CODE_TO'},inplace=True)

depots_dist.drop(depots_dist.index[-1],inplace=True)
depots_dist.drop(depots_dist.index[-1],inplace=True)

for i in range(len(depots_dist)):
    if depots_dist.at[i,'DIRECTION']=='DEPOT->CUSTOMER':
        depots_dist.at[i,'CUSTOMER_CODE_FROM']=0
    else:
        depots_dist.at[i,'CUSTOMER_CODE_FROM']=depots_dist.at[i,'CUSTOMER_CODE_TO']
        depots_dist.at[i,'CUSTOMER_CODE_TO']=0

depots_dist.drop(['DIRECTION','CUSTOMER_NUMBER'],axis=1,inplace=True)
all_dist=pd.concat([customers_dist,depots_dist],ignore_index=True)

all_dist['CUSTOMER_CODE_FROM']=all_dist['CUSTOMER_CODE_FROM'].astype(int)
all_dist['CUSTOMER_CODE_TO']=all_dist['CUSTOMER_CODE_TO'].astype(int)

# process the constraints data
constraints.drop(constraints[constraints['SDVRP_CONSTRAINT_CUSTOMER_CODE']=='139007-1'].index,inplace=True)
constraints.drop_duplicates(subset=['SDVRP_CONSTRAINT_CUSTOMER_CODE','SDVRP_CONSTRAINT_VEHICLE_CODE'],keep='first',inplace=True)
constraints['SDVRP_CONSTRAINT_CUSTOMER_CODE']=constraints['SDVRP_CONSTRAINT_CUSTOMER_CODE'].astype(int)


def u_find(cust_from,cust_to):
    target=all_dist.loc[all_dist['CUSTOMER_CODE_FROM']==cust_from].loc[all_dist.loc[all_dist['CUSTOMER_CODE_FROM']==cust_from]['CUSTOMER_CODE_TO']==cust_to]
    if len(target)>0:
        return target['DISTANCE_KM'].iloc[0],target['TIME_DISTANCE_MIN'].iloc[0]
    else: 
        return -1,-1

def u_find2(cust_from,cust_to):
    if cust_from==0 and cust_to==0:
        return 0,0
    target=all_dist[all_dist['CUSTOMER_CODE_FROM']==cust_from][all_dist[all_dist['CUSTOMER_CODE_FROM']==cust_from]['CUSTOMER_CODE_TO']==cust_to]
    if len(target)>0:
        return target['DISTANCE_KM'].iloc[0],target['TIME_DISTANCE_MIN'].iloc[0]
    else: 
        return -1,-1

