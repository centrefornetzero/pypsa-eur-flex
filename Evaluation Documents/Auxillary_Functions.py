# Packages
#Packages 
import pypsa
import matplotlib.pyplot as plt 
import cartopy 
import geopandas
import networkx
import linopy
import cartopy.crs as ccrs
import atlite 
import geopandas as gpd 
import xarray
import pandas as pd 
from datetime import datetime
import numpy as np
from pypsa.plot import add_legend_patches
import random

## Overall view of colors matched to carriers 
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
tech_colors = {
    'onwind': "#235ebc",
    'onshore wind': "#235ebc",
    'offwind': "#6895dd",
    'offshore wind': "#6895dd",
    'offwind-ac': "#6895dd",
    'offshore_wind (AC)': "#6895dd",
    'offshore wind ac': "#6895dd",
    'offwind-dc': "#74c6f2",
    'offshore wind (DC)': "#74c6f2",
    'offshore wind dc': "#74c6f2",
    'offwind-float': "#b5e2fa",
    'offshore wind (Float)': "#b5e2fa",
    'offshore wind float': "#b5e2fa",
    'hydro': '#298c81',
    'hydro reservoir': '#298c81',
    'ror': '#3dbfb0',
    'run of river': '#3dbfb0',
    'hydroelectricity': '#298c81',
    'PHS': '#51dbcc',
    'hydro+PHS': "#08ad97",
    'solar': "#f9d002",
    'solar PV': "#f9d002",
    'solar-hsat': "#fdb915",
    'solar thermal': '#ffbf2b',
    'residential rural solar thermal': '#f1c069',
    'services rural solar thermal': '#eabf61',
    'residential urban decentral solar thermal': '#e5bc5a',
    'services urban decentral solar thermal': '#dfb953',
    'urban central solar thermal': '#d7b24c',
    'solar rooftop': '#ffea80',
    'OCGT': '#e0986c',
    'OCGT marginal': '#e0986c',
    'OCGT-heat': '#e0986c',
    'gas boiler': '#db6a25',
    'gas boilers': '#db6a25',
    'gas boiler marginal': '#db6a25',
    'residential rural gas boiler': '#d4722e',
    'residential urban decentral gas boiler': '#cb7a36',
    'services rural gas boiler': '#c4813f',
    'services urban decentral gas boiler': '#ba8947',
    'urban central gas boiler': '#b0904f',
    'gas': '#e05b09',
    'fossil gas': '#e05b09',
    'natural gas': '#e05b09',
    'biogas to gas': '#e36311',
    'biogas to gas CC': '#e51245',
    'CCGT': '#a85522',
    'CCGT marginal': '#a85522',
    'allam': '#B98F76',
    'gas for industry co2 to atmosphere': '#692e0a',
    'gas for industry co2 to stored': '#8a3400',
    'gas for industry': '#853403',
    'gas for industry CC': '#692e0a',
    'gas pipeline': '#ebbca0',
    'gas pipeline new': '#a87c62',
    'oil': '#c9c9c9',
    'oil primary': '#d2d2d2',
    'oil refining': '#e6e6e6',
    'imported oil': '#a3a3a3',
    'oil boiler': '#adadad',
    'residential rural oil boiler': '#a9a9a9',
    'services rural oil boiler': '#a5a5a5',
    'residential urban decentral oil boiler': '#a1a1a1',
    'urban central oil boiler': '#9d9d9d',
    'services urban decentral oil boiler': '#999999',
    'agriculture machinery oil': '#949494',
    'shipping oil': "#808080",
    'land transport oil': '#afafaf',
    'Nuclear': '#ff8c00',
    'Nuclear marginal': '#ff8c00',
    'nuclear': '#ff8c00',
    'uranium': '#ff8c00',
    'Coal': '#545454',
    'coal': '#545454',
    'Coal marginal': '#545454',
    'coal for industry': '#343434',
    'solid': '#545454',
    'Lignite': '#826837',
    'lignite': '#826837',
    'Lignite marginal': '#826837',
    'biogas': '#e3d37d',
    'biomass': '#baa741',
    'solid biomass': '#baa741',
    'municipal solid waste': '#91ba41',
    'solid biomass import': '#d5ca8d',
    'solid biomass transport': '#baa741',
    'solid biomass for industry': '#7a6d26',
    'solid biomass for industry CC': '#47411c',
    'solid biomass for industry co2 from atmosphere': '#736412',
    'solid biomass for industry co2 to stored': '#47411c',
    'urban central solid biomass CHP': '#9d9042',
    'urban central solid biomass CHP CC': '#6c5d28',
    'biomass boiler': '#8A9A5B',
    'residential rural biomass boiler': '#a1a066',
    'residential urban decentral biomass boiler': '#b0b87b',
    'services rural biomass boiler': '#c6cf98',
    'services urban decentral biomass boiler': '#dde5b5',
    'biomass to liquid': '#32CD32',
    'unsustainable solid biomass': '#998622',
    'unsustainable bioliquids': '#32CD32',
    'electrobiofuels': '#FF0000',
    'BioSNG': '#123456',
    'solid biomass to hydrogen': '#654321',
    'lines': '#6c9459',
    'transmission lines': '#6c9459',
    'electricity distribution grid': '#97ad8c',
    'low voltage': '#97ad8c',
    'Electric load': '#110d63',
    'electric demand': '#110d63',
    'electricity': '#110d63',
    'industry electricity': '#2d2a66',
    'industry new electricity': '#2d2a66',
    'agriculture electricity': '#494778',
    'battery': '#ace37f',
    'battery storage': '#ace37f',
    'battery charger': '#88a75b',
    'battery discharger': '#5d4e29',
    'home battery': '#80c944',
    'home battery storage': '#80c944',
    'home battery charger': '#5e8032',
    'home battery discharger': '#3c5221',
    'BEV charger': '#baf238',
    'V2G': '#e5ffa8',
    'land transport EV': '#baf238',
    'land transport demand': '#38baf2',
    'EV battery': '#baf238',
    'water tanks': '#e69487',
    'residential rural water tanks': '#f7b7a3',
    'services rural water tanks': '#f3afa3',
    'residential urban decentral water tanks': '#f2b2a3',
    'services urban decentral water tanks': '#f1b4a4',
    'urban central water tanks': '#e9977d',
    'hot water storage': '#e69487',
    'hot water charging': '#e8998b',
    'urban central water tanks charger': '#b57a67',
    'residential rural water tanks charger': '#b4887c',
    'residential urban decentral water tanks charger': '#b39995',
    'services rural water tanks charger': '#b3abb0',
    'services urban decentral water tanks charger': '#b3becc',
    'hot water discharging': '#e99c8e',
    'urban central water tanks discharger': '#b9816e',
    'residential rural water tanks discharger': '#ba9685',
    'residential urban decentral water tanks discharger': '#baac9e',
    'services rural water tanks discharger': '#bbc2b8',
    'services urban decentral water tanks discharger': '#bdd8d3',
    'Heat load': '#cc1f1f',
    'heat': '#cc1f1f',
    'heat vent': '#aa3344',
    'heat demand': '#cc1f1f',
    'rural heat': '#ff5c5c',
    'residential rural heat': '#ff7c7c',
    'services rural heat': '#ff9c9c',
    'central heat': '#cc1f1f',
    'urban central heat': '#d15959',
    'urban central heat vent': '#a74747',
    'decentral heat': '#750606',
    'residential urban decentral heat': '#a33c3c',
    'services urban decentral heat': '#cc1f1f',
    'low-temperature heat for industry': '#8f2727',
    'process heat': '#ff0000',
    'agriculture heat': '#d9a5a5',
    'heat pumps': '#2fb537',
    'heat pump': '#2fb537',
    'air heat pump': '#36eb41',
    'residential urban decentral air heat pump': '#48f74f',
    'services urban decentral air heat pump': '#5af95d',
    'services rural air heat pump': '#5af95d',
    'urban central air heat pump': '#6cfb6b',
    'urban central geothermal heat pump': '#4f2144',
    'geothermal heat pump': '#4f2144',
    'geothermal heat direct utilisation': '#ba91b1',
    'ground heat pump': '#2fb537',
    'residential rural ground heat pump': '#4f2144',
    'residential rural air heat pump': '#48f74f',
    'services rural ground heat pump': '#5af95d',
    'Ambient': '#98eb9d',
    'CHP': '#8a5751',
    'urban central gas CHP': '#8d5e56',
    'CHP CC': '#634643',
    'urban central gas CHP CC': '#6e4e4c',
    'CHP heat': '#8a5751',
    'CHP electric': '#8a5751',
    'district heating': '#e8beac',
    'resistive heater': '#d8f9b8',
    'residential rural resistive heater': '#bef5b5',
    'residential urban decentral resistive heater': '#b2f1a9',
    'services rural resistive heater': '#a5ed9d',
    'services urban decentral resistive heater': '#98e991',
    'urban central resistive heater': '#8cdf85',
    'retrofitting': '#8487e8',
    'building retrofitting': '#8487e8',
    'H2 for industry': "#f073da",
    'H2 for shipping': "#ebaee0",
    'H2': '#bf13a0',
    'hydrogen': '#bf13a0',
    'retrofitted H2 boiler': '#e5a0d9',
    'SMR': '#870c71',
    'SMR CC': '#4f1745',
    'H2 liquefaction': '#d647bd',
    'hydrogen storage': '#bf13a0',
    'H2 Store': '#bf13a0',
    'H2 storage': '#bf13a0',
    'land transport fuel cell': '#6b3161',
    'H2 pipeline': '#f081dc',
    'H2 pipeline retrofitted': '#ba99b5',
    'H2 Fuel Cell': '#c251ae',
    'H2 fuel cell': '#c251ae',
    'H2 turbine': '#991f83',
    'H2 Electrolysis': '#ff29d9',
    'H2 electrolysis': '#ff29d9',
    'NH3': '#46caf0',
    'ammonia': '#46caf0',
    'ammonia store': '#00ace0',
    'ammonia cracker': '#87d0e6',
    'Haber-Bosch': '#076987',
    'Sabatier': '#9850ad',
    'methanation': '#c44ce6',
    'methane': '#c44ce6',
    'Fischer-Tropsch': '#25c49a',
    'liquid': '#25c49a',
    'kerosene for aviation': '#a1ffe6',
    'naphtha for industry': '#57ebc4',
    'methanol-to-kerosene': '#C98468',
    'methanol-to-olefins/aromatics': '#FFA07A',
    'Methanol steam reforming': '#FFBF00',
    'Methanol steam reforming CC': '#A2EA8A',
    'methanolisation': '#00FFBF',
    'biomass-to-methanol': '#EAD28A',
    'biomass-to-methanol CC': '#EADBAD',
    'allam methanol': '#B98F76',
    'CCGT methanol': '#B98F76',
    'CCGT methanol CC': '#B98F76',
    'OCGT methanol': '#B98F76',
    'methanol': '#FF7B00',
    'methanol transport': '#FF7B00',
    'shipping methanol': '#468c8b',
    'industry methanol': '#468c8b',
    'CC': '#f29dae',
    'CCS': '#f29dae',
    'CO2 sequestration': '#f29dae',
    'DAC': '#ff5270',
    'co2 stored': '#f2385a',
    'co2 sequestered': '#f2682f',
    'co2': '#f29dae',
    'co2 vent': '#ffd4dc',
    'CO2 pipeline': '#f5627f',
    'process emissions CC': '#000000',
    'process emissions': '#222222',
    'process emissions to stored': '#444444',
    'process emissions to atmosphere': '#888888',
    'oil emissions': '#aaaaaa',
    'shipping oil emissions': "#555555",
    'shipping methanol emissions': '#666666',
    'land transport oil emissions': '#777777',
    'agriculture machinery oil emissions': '#333333',
    'shipping': '#03a2ff',
    'power-to-heat': '#2fb537',
    'power-to-gas': '#c44ce6',
    'power-to-H2': '#ff29d9',
    'power-to-liquid': '#25c49a',
    'gas-to-power/heat': '#ee8340',
    'waste': '#e3d37d',
    'other': '#000000',
    'geothermal': '#ba91b1',
    'geothermal heat': '#ba91b1',
    'geothermal district heat': '#d19D00',
    'geothermal organic rankine cycle': '#ffbf00',
    'AC': "#70af1d",
    'AC-AC': "#70af1d",
    'AC line': "#70af1d",
    'links': "#8a1caf",
    'HVDC links': "#8a1caf",
    'DC': "#8a1caf",
    'DC-DC': "#8a1caf",
    'DC link': "#8a1caf",
    'load': "#dd2e23",
    'waste CHP': '#e3d37d',
    'waste CHP CC': '#e3d3ff',
    'HVC to air': '#000000',
}




def hex_to_rgba(hex_color, alpha=1.0): # Function that converts color format for later plotting
    """Convert hex color to normalized RGBA tuple (0-1 values)."""
    hex_color = hex_color.lstrip('#')
    if len(hex_color) == 6:
        rgb = tuple(int(hex_color[i:i+2], 16)/255 for i in (0, 2, 4))
        return (*rgb, alpha)
    raise ValueError(f"Invalid hex color: {hex_color}")


def carbon_emissions(network): # emissions generated in the model by generators or in the atmospheric
    carbon_emissions_generators = (network.snapshot_weightings.generators @ network.generators_t.p) / network.generators.efficiency * network.generators.carrier.map(network.carriers.co2_emissions) #Units: MWh / (efficency if thermal and not electric (MWh_el/ MWh_th) ??) * tonnes CO2/MWh
    # This is tonnes co2 emitted from the generators
    carbon_emissions_atmosphere = network.stores_t.e['co2 atmosphere'][-1]
    # the c02 atmosphere store inherits the energy carrier from the bus 
    # the unit of the co2 atmosphere bus is t_co2, so I'm assuming here that
    #  the store input is also ton co2
    # the efficiency2 of the links between the conventional carriers are in 
    # tCO2/MWh (usually MWh_th), so yes, it is entering the stores in tCO2
    return (carbon_emissions_generators, carbon_emissions_atmosphere)


# emissions generated in the model by generators or in the atmospheric

# this function aids the demand function 
# seperate the different types of demand (by carrier)


def load_dataframe_compiler(network, sector_suffix): 
    # this still returns load dataframes in terms of power dispatch, not in 
    # total energy (so MW, not MWH)

    import pandas as pd
    import numpy as np

    # Create a dummy dataframe with load dispatch
    df = pd.DataFrame(network.loads_t.p.T.copy())

    # All loads and their static information 
    df_loads = network.loads.copy()
    carrier_options = df_loads.carrier.unique()

    # Prepare suffix list and remap 'electricity' to empty string
    suffixes = carrier_options.copy()
    if 'electricity' in suffixes:
        suffixes[np.where(suffixes == 'electricity')[0][0]] = ''

    # Check that the given sector_suffix is valid
    if sector_suffix not in suffixes:
        print('network suffix not in suffix list')
        return None

    # Filtering logic
    if sector_suffix == '':
        # Keep only rows where the index suffix after first 5 chars is exactly ''
        filtered_df = df[df.index.str[5:] == ''].copy()
    else:
        # Match any index containing the suffix
        filtered_df = df[df.index.str.contains(sector_suffix, regex=False)].copy()

    return filtered_df
# tested this to make sure that it capture all the loads 
# but for example luxembourg had a negative load dispatch? 


def carrier_totals_sector(network, sector_option):
    # sector option, i.e. -- what form is the energy in before it gets used by a load? 
    # options include: electric, gas -- but this gas must be gas that does not go to the [country]0 buses, 
    # heating (I am guessing here that there are not generators attached to the heating buses, only links, so can use links as a proxy for generation (arrived power/efficiency == original generation))

    import pandas as pd 
    # getting all generation in terms of MWh
    carrier_totals_year = network.generators_t.p.T.groupby([network.generators.carrier, network.generators.bus]).sum().copy()
    carrier_totals_year = carrier_totals_year * network.snapshot_weightings.objective[0]

    # filtering if you want all generation making it's way to the low voltage network, where the residential, agricultural electricity, industrial electricity, and one connection away from EV battery 
    if sector_option == 'electric':
        bus_filter = network.buses[network.buses.unit == 'MWh_el'].index
        mask1 = carrier_totals_year.index.get_level_values(1).isin(bus_filter)

        carrier_totals_year_filtered = carrier_totals_year[carrier_totals_year.index.get_level_values(1).isin(bus_filter)] #in Mwh
        carrier_totals_year_filtered = carrier_totals_year_filtered.sum(axis=1)
        carrier_totals_year_filtered = carrier_totals_year_filtered.groupby('carrier').sum()
        #for all generation that comes from links from conventional carriers
        link_addition = network.links_t.p1.T.groupby([network.links.carrier, network.links.bus1]).sum().copy() * network.snapshot_weightings.objective[0] #end of the link, so it's in MWh electric 
        link_addition = link_addition[link_addition.index.get_level_values(1).isin(bus_filter)]
        carrier_filter = ['coal', 'lignite', 'CCGT', 'OCGT'] #additional bus filter for links to limit to conventional generators
        link_addition = link_addition[link_addition.index.get_level_values(0).isin(carrier_filter) ]
        link_addition = - link_addition.groupby('carrier').sum()
        link_addition = link_addition.sum(axis=1)
        carrier_totals_year_filtered = pd.concat([carrier_totals_year_filtered, link_addition])\
        
    # what are all the links going from a conventional generator to the low voltage loads?
    # n_test.links[n_test.links.bus0 == 'EU lignite']
    # n_test.links[n_test.links.bus0 == 'AL0 0']
    # ugh, no way to disentangle coal or lignite from each other in the al0 to al0 0 low voltage grid, have to say how much is transmitted to the AC level (but that's true for every generator except solar rooftop)
    # still saying, how much MWh_el did you get from coal... it's just that that Mwh_el might be going to the battery, low voltage, or H3

    return carrier_totals_year_filtered


def Entso_data():
    #creating entsoe data for all the countries and arranging them into a multi-indexed series for coparison with network
    #the current csv filepath leads to the 2019 entos historical generation data saved on my local machine 
    ## Buildig in the ENTSO data 
    Entso_generation_2019 = pd.read_csv('/Users/katherine.shaw/Desktop/Data Copies /2019_ENTSO-E_Stats/historical_electricity_production.csv')
    Entso_generation_2019 = Entso_generation_2019.set_index(Entso_generation_2019['Unnamed: 0'], drop = True)
    Entso_generation_2019

    countries = ['AL', 'AT', 'BA', 'BE', 'BG', 'CH', 'CZ', 'DE', 'DK', 'EE', 'ES', 'FI', 'FR', 'GB', 'GR', 'HR', 'HU', 'IE', 'IT', 'LT', 'LU', 'LV', 'ME', 'MK', 'NL', 'NO', 'PL', 'PT', 'RO', 'RS', 'SE', 'SI', 'SK', 'XK']
    Entso_generation_2019_dictionary = dict.fromkeys(countries, 0)  # All keys have value 0
    Entso_generation_2019_dictionary 

    for i in countries:
        country_choice = i 
        # Filter columns that start with the prefix
        filtered_df = Entso_generation_2019[[col for col in Entso_generation_2019.columns if col.startswith(country_choice)]] #filters to just one country by the first two letters of the country code
        filtered_df = filtered_df.drop(filtered_df.index[-1]) #sum of all the rows except the first , as the first gives the carrier
        filtered_df

    #creating the carrier index from the colum names  
        column_list = filtered_df.iloc[0].tolist()
        column_list
    #setting the country as the title of the dictionary? 
        new_dict = dict.fromkeys(column_list, 0)  # All keys have value 0

        if not column_list:
            Entso_generation_2019_dictionary[country_choice] = new_dict
        else: 
            for i in column_list:
             #finding the sum of the column from the original dataframe, 
                carrier_choice = i 
                matching_cols = filtered_df.columns[filtered_df.iloc[0] == carrier_choice]
                gen_column = filtered_df[matching_cols].iloc[1:-1] #this basically turned it into a series
                gen_column = gen_column.astype(float)
                carrier_year_value = gen_column.sum()
                new_dict[carrier_choice] = carrier_year_value.item()
                Entso_generation_2019_dictionary[country_choice] = new_dict
        Entso_generation_2019_dictionary #This is in MWh over the course of the full year 


    series = pd.Series(Entso_generation_2019_dictionary)
    

    # Flatten into a MultiIndex Series
    multi_index_series = pd.Series({(outer, inner): val for outer, inner_dict in series.items() for inner, val in inner_dict.items()})
    multi_index_series

    #The Entso-E data sorted into a multi-level index and then grouped by carrier
    multi_index_series.groupby(level = 1).sum().sort_values(ascending=True)
    multi_index_series.index = pd.MultiIndex.from_tuples(
        [(str(i).lower(), str(j).lower()) for i, j in multi_index_series.index]
    )
    year_Entso_E_series_totals = multi_index_series.groupby(level = 1).sum().sort_values(ascending=True).copy()
    year_Entso_E_series_totals

    #changing the index names to match with the names generated by the pypsa model
    #run of river
    year_Entso_E_series_totals['ror'] = year_Entso_E_series_totals['run of river']
    year_Entso_E_series_totals = year_Entso_E_series_totals.drop('run of river')
    #onshore wind 
    year_Entso_E_series_totals['onwind'] = year_Entso_E_series_totals['onshore wind']
    year_Entso_E_series_totals = year_Entso_E_series_totals.drop('onshore wind')
    year_Entso_E_series_totals
    #offshore wind 
    year_Entso_E_series_totals['offwind'] = year_Entso_E_series_totals['offshore wind']
    year_Entso_E_series_totals = year_Entso_E_series_totals.drop('offshore wind')
    
    #biomass 
    year_Entso_E_series_totals['solid biomass'] = year_Entso_E_series_totals['biomass']
    year_Entso_E_series_totals = year_Entso_E_series_totals.drop('biomass')
    #


    return year_Entso_E_series_totals


def Entso_comparison(network, year): 

    # loading pypsa series and changing into the same index names as enstoe e
    AC_carrier_year = carrier_totals_sector(network, 'electric')
    AC_carrier_year['offwind'] = AC_carrier_year['offwind-ac'] + AC_carrier_year['offwind-dc'] + AC_carrier_year['offwind-float']
    AC_carrier_year = AC_carrier_year.drop(['offwind-ac', 'offwind-dc', 'offwind-float'])
    AC_carrier_year['gas'] = AC_carrier_year['CCGT'] + AC_carrier_year['OCGT']
    AC_carrier_year = AC_carrier_year.drop(['CCGT', 'OCGT'])    

    # Getting entso series 
    year_Entso_E_series_totals = Entso_data()
    
    ######## Making comparison dataframe ######

    # Establishing dataframe 
    comparison_dataframe = pd.DataFrame()

    comparison_dataframe['PyPSA Generation By Carrier'] = AC_carrier_year 
    comparison_dataframe['Entso-E Generation totals 2019'] = year_Entso_E_series_totals
    # delta between the two, 
    comparison_dataframe['delta'] = comparison_dataframe['PyPSA Generation By Carrier'] - comparison_dataframe['Entso-E Generation totals 2019']

    # Plot the horizontal bar chart
    fig, ax = plt.subplots(figsize=(8, 4))
    comparison_dataframe['Entso-E Generation totals 2019'].plot.barh(color='navajowhite', label ='Entso_generation_2019', position = 0.5, )

    # pypsa_bars = comparison_dataframe['PyPSA Generation By Carrier'].plot.barh(color = 'blue', label = 'Pypsa Generation by Carrier 2019', position = 0.5, alpha = 0.5)
    # Add labels to PyPSA bars
    for i, (value, label) in enumerate(zip(comparison_dataframe['PyPSA Generation By Carrier'], 
                                         comparison_dataframe['delta'])):
     ax.text(
        value + 1e7,
        i + 0.2,  # shift down slightly to match bar position
        f"{label:.2e}",
        va='center',
        fontsize=10,
        color='black',
        clip_on=False
    )

    # Optional: extend x-axis limit so labels aren't cut off
    ax.set_xlim(right= 1.5e9)


    plt.title('Comparison along like carriers from the ENTSO-E Transparency Platform and the solved network, 2019 year')
    plt.legend()
    plt.xlabel('MWh')
    plt.tight_layout()

    plt.show()

    return comparison_dataframe


# Create an output of statistics for the model
def model_statistics(n):
    #Total Cost
    system_cost = "{:e}".format(n.objective)
    print('The total system cost as defined by the objective function is :   ')
    print(str(system_cost) + '  EUR')
    print()

    #number of nodes 
    print('The number of nodes is :'  + str(len(n.buses[n.buses['carrier'] == 'AC'])))  
    print()

    print('The carriers with CO2 emissions registered by the model are :  ')
    n_carriers_c02 = n.carriers[n.carriers['co2_emissions'] != 0.0000]
    print(n_carriers_c02.index.values)
    print()
    print('Answer the question: are all the conventional carriers associated with a non-zero carrier carbon emission values?')
    print()
    #The total amount of line length 
    print("The total length of AC lines and DC links in the model is:")
    DC_links = n.links[n.links.carrier == 'DC']
    print(str((n.lines.length.sum() + DC_links.length.sum())) + '  km')
    print()
    
    #the amount of energy curtailed in the model 
    print()
    print('The amount of energy curtailed in the model by carrier is : ')
    print(n.statistics.curtailment())
    print()
    print('Therefore the total amount curtailed is  : ' + str("{:e}".format(n.statistics.curtailment().sum())) + '   MWh per year')
    print()

    #emissions generated in the model

    carbon_emissions1 = carbon_emissions(n)[0]
    print('The carbon emissions registered by the model from generator are ' + str("{:e}".format(carbon_emissions1.sum())) +  '   tonnes CO2 per year')
    carbon_emissions2 = carbon_emissions(n)[1]
    print('The carbon emissions registered by the model in the co2_atmosphere store  ' + str("{:e}".format(-carbon_emissions2)) + '   tonnes CO2')
    print()
 

    #Generators in the model 
    generator_carriers = n.generators.carrier.unique()
    print('The generators types included in this model are :  ')
    print(generator_carriers)
    print()

    #Carriers in the model 
    carrier_types = n.carriers.head(50)
    print('The carriers included in this model are :   ')
    print(n.carriers.index.values)
    print()

    #The demand across the sectors
    n.loads_t.p.columns


    #The global constraints on the model 
    print('The global constrains on the model are:')
    print(n.global_constraints)

    return None 


# Print graphs that indicate demand across the model dimensions
def demand_graphs(n): 
    #The demand across the sectors
    n.loads_t.p.T.index[0:30] 
    load_index = n.loads_t.p.T.index

    suffixes = n.loads.carrier.unique()
    if 'electricity' in suffixes:
        suffixes[np.where(suffixes == 'electricity')[0][0]] = ''

    # Create a dummy dataframe
    df = pd.DataFrame(n.loads_t.p.T.copy())
    # Your desired suffix

    for k in suffixes:
        if k == '': 
            electric_loads = load_dataframe_compiler(n, k)
        if k == 'H2 for industry': 
            H2_loads = load_dataframe_compiler(n, k)
        if k == 'agriculture electricity':
            agriculture_electricity_loads = load_dataframe_compiler(n, k)
        if k == 'agriculture heat':
            agricultural_heat_loads = load_dataframe_compiler(n, k)
        if k == 'gas for industry':
            gas_industry_loads = load_dataframe_compiler(n, k)
        if k == 'industry electricity':
            industry_electric_loads = load_dataframe_compiler(n, k)
        if k == 'land transport EV':
            land_transport_EV_loads = load_dataframe_compiler(n, k)
        if k == 'low-temperature heat for industry':
            low_temp_heat_industry_loads =load_dataframe_compiler(n, k)
        if k == 'rural heat':
            rural_heat_loads = load_dataframe_compiler(n, k)
        if k == 'urban central heat':
            urban_central_heat = load_dataframe_compiler(n, k)
        if k == 'urban decentral heat':
            urban_decentral_heat = load_dataframe_compiler(n, k)
        if k == 'solid biomass for industry':
            solid_biomass_for_industry = load_dataframe_compiler(n,k)
        if k == 'industry methanol':
            industry_methanol = load_dataframe_compiler(n,k)
        if k == 'kerosene for aviation':
            kerosene_for_aviation = load_dataframe_compiler(n,k)
        if k == 'process emissions':
            process_emissions = load_dataframe_compiler(n,k)
        if k == 'agriculture machinery oil':
            agriculture_machinery_oil = load_dataframe_compiler(n,k)
        elif k not in suffixes: 
            print('suffix name not found, likely typo')


    import matplotlib.pyplot as plt
    import inspect

    # List of your dataframes
    series_list = [electric_loads.sum(), H2_loads.sum(), agricultural_heat_loads.sum(), agriculture_electricity_loads.sum(), 
                gas_industry_loads.sum(), industry_electric_loads.sum(), land_transport_EV_loads.sum(), low_temp_heat_industry_loads.sum(), 
                rural_heat_loads.sum(), urban_central_heat.sum(), urban_decentral_heat.sum(),
                solid_biomass_for_industry.sum(), industry_methanol.sum(), kerosene_for_aviation.sum(), process_emissions.sum(), agriculture_machinery_oil.sum() ]

    #list of names (for plot titles)
    series_list_names = ["electric_loads", "H2_loads", 'agricultural_heat_loads', 'agriculture_electricity_loads', 
                'gas_industry_loads', 'industry_electric_loads', 'land_transport_EV_loads', 'low_temp_heat_industry_loads', 
                'rural_heat_loads', 'urban_central_heat', 'urban_decentral_heat',
                'solid_biomass_for_industry', 'industry_methanol', 'kerosene_for_aviation', 'process_emissions', 'agriculture_machinery_oil']


    #urban_decentral_heat.sum().T.div(1e3).plot() #turned MW into GW

    fig, axes = plt.subplots(8, 2, figsize=(8, 20))  # Adjust figsize as needed

    # Flatten axes array for easy iteration
    axes = axes.flatten()

    for i, df in enumerate(series_list):
        ax = axes[i]
        df.div(1e3).plot(ax=ax)  # Assumes df is plottable directly #Turns MW into GW 
        title = series_list
        ax.set_title(series_list_names[i])
        ax.set_ylabel('GW')

    # If you have fewer than 12 DataFrames, hide the unused subplots
    for j in range(len(series_list), len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()

    return None


# creates a pie chart, for things at the high AC level only, uses carrier_totals_sector(network, 'electric') to get the generation totals for each carrier
def carrier_pie_chart(network): #this is for the electric sector only
    import matplotlib.pyplot as plt
    import pandas as pd
    import matplotlib.colors as mcolors
    import numpy as np 
    #nuclear included with conventional genrators here, different across different functions to deal with integration issues
    carrier_seperation_list = carrier_totals_sector(network, 'electric')
    conventional_carriers = ['nuclear', 'oil', 'OCGT', 'CCGT', 'coal', 'lignite', 'geothermal', 'solid biomass', 'biogas']
    renewable_carriers = ['solar', 'solar-hsat', 'onwind', 'offwind-ac', 'offwind-dc', 'offwind-float', 'offshore wind', 'hydro', 'ror', 'solar rooftop']

    conventional_sum = carrier_seperation_list[carrier_seperation_list.index.intersection(conventional_carriers)]
    renewable_sum = carrier_seperation_list[carrier_seperation_list.index.intersection(renewable_carriers)]

    #changing series to dictionaries (to make multi-index series)
    conventional_sum = conventional_sum.to_dict()
    renewable_sum = renewable_sum.to_dict()

    #adding in 'conventional' and 'renewable' for first index 
    conventional_sum = { 'conventional' : conventional_sum}
    renewable_sum = { 'renewable' : renewable_sum}

    #can I create a multi-index series where the first index is 'conventional' or 'renewable', and then the second index is the carrier and the values 
    conventional_sum = pd.Series({(outer, inner): val for outer, inner_dict in conventional_sum.items() for inner, val in inner_dict.items()})
    renewable_sum = pd.Series({(outer, inner): val for outer, inner_dict in renewable_sum.items() for inner, val in inner_dict.items()})
    print(conventional_sum)
    print(renewable_sum)

    # Combine the two series
    s_all = pd.concat([conventional_sum, renewable_sum])

    # Get groups and unique technologies
    groups = s_all.index.get_level_values(0).unique()
    technologies = s_all.index.get_level_values(1).unique()

    # Define base group colors
    group_base_colors = {
        'conventional': 'peru',          # light brown
        'renewable': 'mediumseagreen'    # light green
    }

    # Build consistent color map for each technology using alpha gradation
    tech_color_map = {}
    for group in groups:
        techs_in_group = s_all.loc[group].index.tolist()
        n = len(techs_in_group)
        base_rgba = np.array(mcolors.to_rgba(group_base_colors[group]))

        # Generate different alpha/brightness values
        for i, tech in enumerate(techs_in_group):
            # Slightly darken each successive wedge (adjust factor as needed)
            factor = 0.6 + 0.4 * (1 - i / max(1, n - 1))  # from 1.0 to 0.6
            shaded_rgb = base_rgba[:3] * factor
            tech_color_map[(group, tech)] = np.clip(shaded_rgb, 0, 1)

    # Compute group sums and total
    group_totals = s_all.groupby(level=0).sum()
    total_sum = s_all.sum()

    # Labels with percentages
    outer_labels = [
        f"{tech}\n{value / total_sum:.1%}" 
        for (_, tech), value in s_all.items()
    ]
    inner_labels = [
        f"{group}\n{value / total_sum:.1%}"
        for group, value in group_totals.items()
    ]

    # Colors from the map
    outer_colors = [tech_color_map[key] for key in s_all.index]
    inner_colors = [group_base_colors[group] for group in group_totals.index]

    # Plot
    fig, ax = plt.subplots(figsize=(8, 8))

    # Outer ring: technologies
    ax.pie(
        s_all.values,
        radius=1,
        labels=outer_labels,
        colors=outer_colors,
        labeldistance=1.1,
        wedgeprops=dict(width=0.3, edgecolor='white')
    )

    # Inner ring: groups
    ax.pie(
        group_totals.values,
        radius=0.7,
        labels=inner_labels,
        colors=inner_colors,
        labeldistance=0.55,  # Tighter so labels stay inside the inner ring
        wedgeprops=dict(width=0.3, edgecolor='white')
    )

    ax.set(aspect='equal', title='Energy Mix Breakdown (Shaded Wedges)')
    #plt.show()

    return None


# dispatch of all generators in time (GW) -- this does not include CURTAILED components of any asset
def generator_dispatch(network):
    #nuclear included with conventional generators here (different elsewhere)
    generation_by_carrier = network.generators_t.p.T.groupby(network.generators.carrier).sum()
    renewable_carriers = ['solar', 'solar-hsat', 'onwind', 'offwind-ac', 'offwind-dc', 'offwind-float', 'offshore wind', 'hydro', 'ror', 'solar rooftop']
    conventional_carriers = ['nuclear', 'oil', 'OCGT', 'CCGT', 'coal', 'lignite', 'geothermal', 'biogas', 'gas', 'solid biomass']
    fix, ax = plt.subplots(figsize = (14,6))
    ax.set_ylim(0,11e2)
    #conventional generation dispatch
    conventional_generation = generation_by_carrier[generation_by_carrier.index.isin(conventional_carriers)]
    conventional_generation = conventional_generation.div(1e3)
    fig1 = conventional_generation.T.plot(
        kind = 'area',
        stacked = True,
        cmap = 'tab20b',
        ax = ax,
        )
    plt.ylabel('GW')
    plt.title('Conventional Generation at generator (MWh_th), weather year 2019, not sector specific (i.e. gas can be used for heating)')
    plt.show()

    #renewable generation dispatch
    fix, ax = plt.subplots(figsize = (14,6))
    ax.set_ylim(0,8e2)
    renewable_generation = generation_by_carrier[generation_by_carrier.index.isin(renewable_carriers)]
    renewable_generation = renewable_generation.div(1e3) #turned MW into Gw
    fig2 = renewable_generation.T.plot(
        kind = 'area',
        stacked = True,
        cmap = 'tab20b',
        ax = ax,
        )
    plt.ylabel('GW')
    plt.title('Renewable Generation, weather year 2019')
    plt.show()

    #total generation dispatch (THIS IS MIXING AND MATCHING UNITS, MW_TH VS MW_EL, FOR TOTAL DISPATCH LOOK AT THE FUNCTION DISPATCH AT NODES)

    return fig1


# This is different from the above because the generator's dispatch is measured AT the POINT OF USE (MWh_el nodes), rather than at the point of generation
def dispatch_at_nodes(network): 
    generation_by_carrier = network.generators_t.p.T.groupby(network.generators.carrier).sum()
    renewable_carriers = ['solar', 'solar-hsat', 'onwind', 'offwind-ac', 'offwind-dc', 'offwind-float', 'offshore wind', 'hydro', 'ror', 'solar rooftop', 'nuclear']
    conventional_carriers = [ 'oil', 'OCGT', 'CCGT', 'coal', 'lignite', 'geothermal', 'biogas', 'gas', 'solid biomass']

    #renewable dispatch 
    fig, ax = plt.subplots(figsize = (14,6))
    ax.set_ylim(0,8e2)
    renewable_generation = generation_by_carrier[generation_by_carrier.index.isin(renewable_carriers)]
    renewable_generation = renewable_generation.div(1e3) #turned MW into Gw
    fig2 = renewable_generation.T.plot(
        kind = 'area',
        stacked = True,
        cmap = 'tab20b',
        ax = ax,
        )
    plt.ylabel('GW')
    plt.title('Renewable Generation, weather year 2019')
    plt.show()

    #conventional dispatch going to electric nodes (nodes where the bus has units of MWh_el)
    fig, ax = plt.subplots(figsize = (14,6))
    buses_with_conventional_carriers = network.buses.copy()[network.buses.carrier.isin(conventional_carriers)].index
    conventional_links = network.links[network.links.bus0.isin(buses_with_conventional_carriers)]
    #want to separate the conventionals that are going to buses that operate in Mwh_th and buses that operate in MWh_el
    Mwh_el_buses = network.buses[network.buses.unit == 'MWh_el'].index
    MWh_th_buses = network.buses[network.buses.unit == 'MWh_th'].index
    conventioanl_links_from_thermal_to_electric = conventional_links[conventional_links.bus1.isin(Mwh_el_buses)]
    conventioanl_links_from_thermal_to_thermal = conventional_links[conventional_links.bus1.isin(MWh_th_buses)]
    
    #conventional thermal to electric dispatch
    mask = conventioanl_links_from_thermal_to_electric.index
    conventional_to_electric_dispatch_at_end_of_link = ((network.links_t.p1.T.loc[mask]).groupby(network.links.carrier).sum().T)*-1
    conventional_to_electric_dispatch_at_end_of_link.div(1e3).plot(
        kind = 'area',
        stacked = True,
        cmap = 'tab20b',
        ax = ax,
    )
    plt.legend(loc = 'upper right')
    plt.ylabel('GW')
    plt.title('Conventional Generation Links to MWh_el buses, abs(p1) value at end of link')
    plt.show()
    
    #Renewable and conventional going to electric nodes (together)
    fig, ax = plt.subplots(figsize = (14,6))
    ax.set_ylim(-3e2,8e2)
    renewable_generation = renewable_generation.T
    total_generation = (-conventional_to_electric_dispatch_at_end_of_link.div(1e3)).join(renewable_generation) #negative only for graphhing purposes, the actual dispatc is TO the node, not taking from it
    total_generation.plot(
        kind = 'area',
        stacked = True,
        cmap = 'tab20b',
        ax = ax
    )
    plt.legend(loc = 'upper right')
    plt.ylabel('GW')
    plt.title('Total Generation delivered to electric nodes')
    plt.show()

    #conventional dispatch going to the thermal nodes (mostly heat)
    #conventional thermal to thermal dispatch
    fig, ax = plt.subplots(figsize = (14,6))
    mask = conventioanl_links_from_thermal_to_thermal.index
    conventional_to_thermal_dispatch_at_end_of_link = ((network.links_t.p1.T.loc[mask]).groupby(network.links.carrier).sum().T)*-1
    conventional_to_thermal_dispatch_at_end_of_link.div(1e3).plot(
        kind = 'area',
        stacked = True,
        cmap = 'tab20b',
        ax = ax,
    )
    plt.legend(loc = 'upper right')
    plt.ylabel('GW')
    plt.title('Conventional Generation Links to MWh_th buses, abs(p1) value at end of link')



    return None


# figures of stores and their levels over the time period 
def plot_store_energy_by_carrier(network, figsize=(15, 10)):
    """
    Plots time series of total energy in storage for each carrier type.

    Parameters
    ----------
    network : pypsa.Network
        The PyPSA network object.
    figsize : tuple
        Size of the entire figure.
    """
    # Get all store carriers
    store_carriers = network.stores.carrier.unique()

    # Prepare the plot
    num_carriers = len(store_carriers)
    fig, axes = plt.subplots(num_carriers, 1, figsize=figsize, sharex=True)

    if num_carriers == 1:
        axes = [axes]  # Make iterable if only one plot

    for ax, carrier in zip(axes, store_carriers):
        # Select stores with the current carrier
        store_subset = network.stores[network.stores.carrier == carrier]
        store_ids = store_subset.index

        # Sum the energy across all stores of this carrier over time
        energy_sum = network.stores_t.e[store_ids].sum(axis=1)

        # Determine the unit(s) of the buses the stores are connected to
        store_buses = store_subset['bus']
        bus_units = network.buses.loc[store_buses, 'unit'].dropna().unique()

        # Dynamic y-label based on unit
        if len(bus_units) == 1:
            ylabel = f"Energy [{bus_units[0]}]"
        elif len(bus_units) > 1:
            ylabel = "Energy [" + ", ".join(bus_units) + "]"
        else:
            ylabel = "Energy [MWh]"  # fallback if unit is missing

        # Plot
        energy_sum.plot(ax=ax)
        ax.set_title(f"{carrier}")
        ax.set_ylabel(ylabel)

    plt.xlabel("Time")
    plt.tight_layout()
    plt.show()

    return None


# dispatch of stores and storage units for the power sector 
def stores_for_power(network):
    import pandas as pd
    import matplotlib.pyplot as plt
    #dispatch of stores and storage units for the power sector 
    Mwh_el_buses = n_test.buses[n_test.buses.unit == 'MWh_el'].index
    MWh_th_buses = n_test.buses[n_test.buses.unit == 'MWh_th'].index  

    electric_stores = network.stores[network.stores.bus.isin(Mwh_el_buses)]
    #print(electric_stores.index)

    electric_store_dispatch = network.stores_t.p.T.loc[electric_stores.index]
    electric_store_dispatch = electric_store_dispatch.groupby(network.stores.carrier).sum().T
    electric_store_dispatch = electric_store_dispatch.div(1e3) #changing Mw to Gw
    fig, ax = plt.subplots(figsize = (14,6))
    fig = electric_store_dispatch.plot(
        ax = ax,
        kind = 'area',
        stacked = False,
        cmap = 'tab20b'
    )
    plt.title('Store Dispatch to Mwh_el buses')
    plt.ylabel('GW')
    plt.show()

    #now for the thermal stores 
    thermal_stores = network.stores[network.stores.bus.isin(MWh_th_buses)]
    thermal_store_dispatch = network.stores_t.p.T.loc[thermal_stores.index]
    thermal_store_dispatch = thermal_store_dispatch.groupby(network.stores.carrier).sum().T
    thermal_store_dispatch = thermal_store_dispatch.div(1e3) #transforming Mw to Gw
    fig, ax = plt.subplots(figsize = (14,6))
    fig = thermal_store_dispatch.plot(
        ax = ax,
        kind = 'area',
        stacked = False,
        cmap = 'tab20b'
    )
    plt.title('Store Dispatch to Mwh_th buses')
    plt.ylabel('GW')
    plt.show()
    #print(thermal_stores.index)
    return fig 


# dispatch of storage units for the power sector 
def storage_units_for_power(network): #all storage units connect to transmiession level buses 
    import pandas as pd
    import matplotlib.pyplot as plt
    #dispatch of stores and storage units for the power sector 
    fig, ax = plt.subplots(figsize = (14,6)) #this is the time where you'd do like a seven day rolling mean
    storage_unit_dispatch = network.storage_units_t.p.T.groupby(n_test.storage_units.carrier).sum().T.div(1e3) #changed MW to Gw
    storage_unit_dispatch.plot(
        ax = ax,
        kind = 'area', 
        stacked = False,
        cmap = 'tab20b')
    plt.title('Store Unit dispatch (they only go to high level transmission buses)')
    plt.ylabel('GW')
    plt.show()
    return None 

# dispatch across stores, storage units, and generators 
def all_dispatch_plus_load_electric(network):
    fig, ax = plt.subplots(figsize=(14,6))
    # generation associations
    generation_by_carrier = network.generators_t.p.T.groupby(network.generators.carrier).sum()
    renewable_carriers = ['solar', 'solar-hsat', 'onwind', 'offwind-ac', 'offwind-dc', 'offwind-float', 'offshore wind', 'hydro', 'ror', 'solar rooftop', 'nuclear']
    conventional_carriers = [ 'oil', 'OCGT', 'CCGT', 'coal', 'lignite', 'geothermal', 'biogas', 'gas', 'solid biomass']

    #renewable dispatch 
    #fig, ax = plt.subplots(figsize = (14,6))
    renewable_generation = generation_by_carrier[generation_by_carrier.index.isin(renewable_carriers)]
    renewable_generation = renewable_generation.div(1e3) #turned MW into Gw
    total_dispatch = renewable_generation.T.copy()

    #conventional links to electric buses (to which electric demand will be connected)
    buses_with_conventional_carriers = network.buses.copy()[network.buses.carrier.isin(conventional_carriers)].index
    conventional_links = network.links[network.links.bus0.isin(buses_with_conventional_carriers)]
    #want to separate the conventionals that are going to buses that operate in Mwh_th and buses that operate in MWh_el
    Mwh_el_buses = network.buses[network.buses.unit == 'MWh_el'].index
    conventioanl_links_from_thermal_to_electric = conventional_links[conventional_links.bus1.isin(Mwh_el_buses)]
    
    #conventional thermal to electric dispatch
    mask = conventioanl_links_from_thermal_to_electric.index
    conventional_to_electric_dispatch_at_end_of_link = ((network.links_t.p1.T.loc[mask]).groupby(network.links.carrier).sum().T)*-1
    total_dispatch = conventional_to_electric_dispatch_at_end_of_link.div(1e3).join(total_dispatch)


    #stores
    electric_stores = network.stores[network.stores.bus.isin(Mwh_el_buses)]
    electric_store_dispatch = network.stores_t.p.T.loc[electric_stores.index]
    electric_store_dispatch = electric_store_dispatch.groupby(network.stores.carrier).sum().T
    electric_store_dispatch = electric_store_dispatch.div(1e3) #changing Mw to Gw
    electric_store_dispatch_positive = electric_store_dispatch.clip(lower=0)
    electric_store_dispatch_charging = electric_store_dispatch.clip(upper=0)
    total_dispatch = electric_store_dispatch_positive.join(total_dispatch)



     #storage units
    electric_storage_units = network.storage_units[network.storage_units.bus.isin(Mwh_el_buses)]
    electric_storage_unit_dispatch = network.storage_units_t.p.T.loc[electric_storage_units.index]
    electric_storage_unit_dispatch = electric_storage_unit_dispatch.groupby(network.storage_units.carrier).sum().T
    electric_storage_unit_dispatch = electric_storage_unit_dispatch.div(1e3) #changing Mw to Gw
    electric_storage_unit_dispatch_positive = electric_storage_unit_dispatch.clip(lower=0)
    electric_storage_unit_dispatch_charging = electric_storage_unit_dispatch.clip(upper=0)
    total_dispatch = electric_storage_unit_dispatch_positive.join(total_dispatch)
    ax.set_ylim(-2e2,15e2)
    #total_generation_graph
    total_dispatch.plot(
        ax = ax,
        kind = 'area',
        stacked = True,
        cmap = 'tab20b'
    )
    electric_storage_unit_dispatch_charging.plot(
        ax = ax,
        kind ='area',
        stacked = False,
        cmap = 'tab20b'
    )
    ax.set_ylim(-2e2,15e2)
    #all loads connected to electric buses 
    Mwh_el_buses = network.buses[network.buses.unit == 'MWh_el'].index
        #the loads connected to electric buses include 'electricity', 'land transport EV', 'industry electricity','agriculture electricity', and do not include
    electric_loads = network.loads[network.loads.bus.isin(Mwh_el_buses)].index
    electric_loads_time = network.loads_t.p.T.loc[electric_loads].div(1e3) #turning MW into GW
    electric_loads_time = electric_loads_time.groupby(network.loads.carrier).sum()
    electric_loads_time = electric_loads_time.T
    #if you want to keep the electricity load types sepeate, comment out the line below
    electric_loads_time = electric_loads_time.sum(axis=1)
        # Apply 3-day rolling mean (72 hours, given 6-hour intervals = 4 time steps per day → window=12)
    electric_loads_rolling = electric_loads_time.rolling(window=12, center=True).mean()
    electric_loads_rolling.plot(
        ax = ax,
        kind = 'line',
        color = 'k'
    )
    plt.title('3-day rolling mean of electricity loads by Carrier')
    plt.ylabel('GW')
    plt.show()
    ax.set_ylim(-2e2, 15e2)  # setting the y limits to be the same as the load graph


    return None 


#plotting generator dispatch at each node (Mwh over the year, not MW)
def generator_topology_plot(network, sector): 
    " the options for the 'sector' argument come from the buses, so we have 'buses_AC', 'buses_none', 'buses_co2', 'buses_co2 stored', "
    "'buses_co2 sequestered', 'buses_gas', 'buses_H2', 'buses_battery', 'buses_EV battery', 'buses_urban central heat', "
    "'buses_urban central water tanks', 'buses_biogas', 'buses_solid biomass', 'buses_methanol', 'buses_low voltage', "
    "'buses_home battery', 'buses_rural heat', 'buses_rural water tanks', 'buses_urban decentral heat', "
    "'buses_urban decentral water tanks']"
    fig, ax = plt.subplots(figsize = (14,10))
    ax=plt.axes(projection = ccrs.EqualEarth())

    bus_types = {}
    for i in network.buses.carrier.unique():
        bus_list = network.buses[network.buses['carrier'] == i ]
        bus_types[f"buses_{i}"] = bus_list

    #I should be making bus size proportional to electric load, because I want to size to be consistent across all the maps
    #though I can't use the load in the bus sizes argument because I want the visual indication of generator profile
    #and that comes from the generation, so I maybe later I will scale the generation values until they are numerically equivalent to the load
    #they wouldn't be automatically because some of the load is satisfied by transmission and storage.

    #s is a series of the generation in MWh multi-indexed by bus and carrier, first index bus, second index carrier

    
    
    #the filtering by sector 
        # electric
    if sector == 'electric':
        s1 = network.generators_t.p.T.mul(network.snapshot_weightings.objective[0]).sum(axis=1).groupby([network.generators.bus, network.generators.carrier]).sum()
        s = s1
        bus_filtered = bus_types['buses_AC'].index
    if sector == 'gas':
        s1 = network.generators_t.p.T.mul(network.snapshot_weightings.objective[0]).sum(axis=1).groupby([network.generators.bus, network.generators.carrier]).sum()
        s = s1        
        bus_filtered = bus_types['buses_gas'].index
    if sector == 'H2':
        s1 = network.generators_t.p.T.mul(network.snapshot_weightings.objective[0]).sum(axis=1).groupby([network.generators.bus, network.generators.carrier]).sum()
        s = s1 
        bus_filtered = bus_types['buses_H2'].index
    if sector == 'battery':
        s1 = network.generators_t.p.T.mul(network.snapshot_weightings.objective[0]).sum(axis=1).groupby([network.generators.bus, network.generators.carrier]).sum()
        s = s1        
        bus_filtered = bus_types['buses_battery'].index
    if sector == 'EV battery' :
        s1 = network.generators_t.p.T.mul(network.snapshot_weightings.objective[0]).sum(axis=1).groupby([network.generators.bus, network.generators.carrier]).sum()
        s = s1
        bus_filtered = bus_types['buses_EV battery'].index
    if sector ==  'low voltage': #for the carrier 'electricity distribution grid' given here is power coming from the AC transmission buses
        s1 = network.generators_t.p.T.mul(network.snapshot_weightings.objective[0]).sum(axis=1).groupby([network.generators.bus, network.generators.carrier]).sum()
        s = s1       
        bus_filtered = bus_types['buses_low voltage'].index
    if sector == 'urban heat':
        s1 = network.generators_t.p.T.mul(network.snapshot_weightings.objective[0]).sum(axis=1).groupby([network.generators.bus, network.generators.carrier]).sum()
        s = s1
        bus_filtered = bus_types['buses_urban central heat'].index
    if sector == 'rural heat':
        s1 = network.generators_t.p.T.mul(network.snapshot_weightings.objective[0]).sum(axis=1).groupby([network.generators.bus, network.generators.carrier]).sum()
        s = s1
        bus_filtered = bus_types['buses_rural heat'].index
    if sector == 'urban decentral heat':
        s1 = network.generators_t.p.T.mul(network.snapshot_weightings.objective[0]).sum(axis=1).groupby([network.generators.bus, network.generators.carrier]).sum()
        s = s1
        bus_filtered = bus_types['buses_urban decentral heat'].index
    if sector == 'water tanks': 
        s1 = network.generators_t.p.T.mul(network.snapshot_weightings.objective[0]).sum(axis=1).groupby([network.generators.bus, network.generators.carrier]).sum()
        s = s1
        bus_filtered = (bus_types['buses_urban decentral water tanks'] + bus_types['buses_urban central water tanks'] + bus_types['buses_rural water tanks']).index 
    if sector == 'biogas':
        s1 = network.generators_t.p.T.mul(network.snapshot_weightings.objective[0]).sum(axis=1).groupby([network.generators.bus, network.generators.carrier]).sum()
        s = s1
        bus_filtered = bus_types['buses_biogas'].index
    if sector == 'solid biomass': #this one graphs wierdly 
        s1 = network.generators_t.p.T.mul(network.snapshot_weightings.objective[0]).sum(axis=1).groupby([network.generators.bus, network.generators.carrier]).sum()
        s = s1
        bus_filtered = bus_types['buses_solid biomass'].index
  
    s_filtered = s.loc[bus_filtered]
    #transparency_value = transparency_value.loc[bus_filtered] if 'transparency_value' in locals() else None


    #scaling s_filtered so that it is numerically equivalent to the node at each bus
    #plotting the graph
    network_figure = network.plot(
        margin = 0.2, 
        ax = ax,
        bus_sizes = s_filtered / (s_filtered.max()),  
        #bus_alpha = round(((abs(transparency_value)/ s_filtered).sum() / len(bus_filtered)),2 ), 
    )

    #Adding the legend 
    
    carrier_plot_list = []
    for i in range(len(s_filtered.index)):
      carrier_plot_list.append(s_filtered.index[i][1])
    carrier_plot_list = pd.Series(carrier_plot_list)
    filter_criteria = carrier_plot_list.unique().copy() #great that looks about right 
    filter_criteria 

    #filter the tech_color dictionary to have just the carriers I want

    filtered_dict = {k: v for k, v in tech_colors.items() if k in filter_criteria}
    filtered_dict

    #turn that dictionary into the axes 
    legend_handles = [Patch(facecolor=color, label=carrier) for carrier, color in filtered_dict.items()]
    # Add the legend
    ax.legend(
        handles=legend_handles,
        title="Carriers",
        loc="center right",
        frameon=False,
        bbox_to_anchor=(0, 0.5),
        ncols = 1,
        prop={'size': 18},           # <-- Change this value to your desired font size
        title_fontsize=18 
    )

    return network_figure

#the only issue with using plot network test where the bus sizes are relative to the largest contirbuting country is that 
#plotting total energy used to satisfy demand at each node (including every component: generator,store,lines and links)
#Function to plot the network on an equal area map -- it's now modified to include all power arriving at the bus, not just the stores and generators associated with the bus
def plot_network_test(network, sector):
    " the options for the 'sector' argument come from the buses, so we have 'buses_AC', 'buses_none', 'buses_co2', 'buses_co2 stored', "
    "'buses_co2 sequestered', 'buses_gas', 'buses_H2', 'buses_battery', 'buses_EV battery', 'buses_urban central heat', "
    "'buses_urban central water tanks', 'buses_biogas', 'buses_solid biomass', 'buses_methanol', 'buses_low voltage', "
    "'buses_home battery', 'buses_rural heat', 'buses_rural water tanks', 'buses_urban decentral heat', "
    "'buses_urban decentral water tanks']"

    from pypsa.plot import add_legend_patches
    import random
    bus_types = {} 
    for i in network.buses.carrier.unique():
        bus_list = network.buses[network.buses['carrier'] == i ]
        bus_types[f"buses_{i}"] = bus_list

    fig = plt.figure(figsize=(20,16))
    ax=plt.axes(projection = ccrs.EqualEarth())


    #s is a series of the generation in MWh multi-indexed by bus and carrier, first index bus, second index carrier
    #This will allow for filtering by sector (sort of) based on bus
    #for sectors where there is interaction with via discharging and charging (H2, battery, EV battery), I have found that the optimizer likes to make these equally charged and discharged, which leads to very very small overall sums
    #To address this, the size of the bus will be correlated to the amount the store is charged, and the opacity of the bus will be correlated to the amount the store is discharged
    #The more opaque the bus, the more it is equally charged to discharged.
    # option in the .plot() section below, uncomment to enable 
    
    # Step 1: Get the weighted power time series
    store_p = network.stores_t.p.T.mul(network.snapshot_weightings.objective[0])  # shape: (store, snapshot)
        
    # Step 2: Add 'bus' and 'carrier' information to store_p
    store_p.index = pd.MultiIndex.from_arrays([network.stores.bus.loc[store_p.index], network.stores.carrier.loc[store_p.index]], names=['bus', 'carrier'])

    # Step 3: Compute sums of positive and negative values separately
    s = store_p[store_p > 0].sum(axis=1)  # sum of positive values for each (bus, carrier) #s will be set to positive sums, then the transparency will be found by using negative sums

    #stores connect directly to each bus, so for sectors like electric, stores do not need to be incorperated as it was checked that no stores of the other carriers connect directly to those buses

    #the filtering by sector 
        # electric
    if sector == 'electric':
        s1 = network.generators_t.p.T.mul(network.snapshot_weightings.objective[0]).sum(axis=1).groupby([network.generators.bus, network.generators.carrier]).sum()
        s2 = network.storage_units_t.p.T.mul(network.snapshot_weightings.objective[0]).sum(axis=1).groupby([network.storage_units.bus, network.storage_units.carrier]).sum()
        s = s1.add(s2, fill_value = 0)
        s3 = network.links_t.p1.T.mul(network.snapshot_weightings.objective[0]).sum(axis=1).groupby([network.links.bus1, network.links.carrier]).sum() #I checked, the DC links that are connected to each bus only represent the power going TO that bus, not the DC power FROM that bus to another
        s3.index = s3.index.rename(['bus'] + list(s3.index.names[1:]))
        s = s.add(abs(s3), fill_value = 0)
        bus_filtered = bus_types['buses_AC'].index
    if sector == 'gas':
        transparency_value = store_p[store_p < 0].sum(axis=1)  # sum of negative values for each (bus, carrier) #will determine the transparency of the bus
        bus_filtered = bus_types['buses_gas'].index
    if sector == 'H2':
        transparency_value = store_p[store_p < 0].sum(axis=1)  # sum of negative values for each (bus, carrier) #will determine the transparency of the bus
        bus_filtered = bus_types['buses_H2'].index
    if sector == 'battery':
        transparency_value = store_p[store_p < 0].sum(axis=1)  # sum of negative values for each (bus, carrier) #will determine the transparency of the bus
        bus_filtered = bus_types['buses_battery'].index
    if sector == 'EV battery' :
        transparency_value = store_p[store_p < 0].sum(axis=1)  # sum of negative values for each (bus, carrier) #will determine the transparency of the bus
        bus_filtered = bus_types['buses_EV battery'].index
    if sector ==  'low voltage': #for the carrier 'electricity distribution grid' given here is power coming from the AC transmission buses
        s1 = network.generators_t.p.T.mul(network.snapshot_weightings.objective[0]).sum(axis=1).groupby([network.generators.bus, network.generators.carrier]).sum()
        s2 = network.storage_units_t.p.T.mul(network.snapshot_weightings.objective[0]).sum(axis=1).groupby([network.storage_units.bus, network.storage_units.carrier]).sum()
        s = s1.add(s2, fill_value = 0) 
        s3 = network.links_t.p1.T.mul(network.snapshot_weightings.objective[0]).sum(axis=1).groupby([network.links.bus1, network.links.carrier]).sum()
        s3.index = s3.index.rename(['bus'] + list(s3.index.names[1:]))
        s = s.add(abs(s3), fill_value = 0)       
        bus_filtered = bus_types['buses_low voltage'].index
    if sector == 'urban heat':
        s1 = network.generators_t.p.T.mul(network.snapshot_weightings.objective[0]).sum(axis=1).groupby([network.generators.bus, network.generators.carrier]).sum()
        s2 = network.storage_units_t.p.T.mul(network.snapshot_weightings.objective[0]).sum(axis=1).groupby([network.storage_units.bus, network.storage_units.carrier]).sum()
        s = s1.add(s2, fill_value = 0)
        transparency_value = None 
        bus_filtered = bus_types['buses_urban central heat'].index
    if sector == 'rural heat':
        s1 = network.generators_t.p.T.mul(network.snapshot_weightings.objective[0]).sum(axis=1).groupby([network.generators.bus, network.generators.carrier]).sum()
        s2 = network.storage_units_t.p.T.mul(network.snapshot_weightings.objective[0]).sum(axis=1).groupby([network.storage_units.bus, network.storage_units.carrier]).sum()
        s = s1.add(s2, fill_value = 0)
        s3 = network.links_t.p1.T.mul(network.snapshot_weightings.objective[0]).sum(axis=1).groupby([network.links.bus1, network.links.carrier]).sum()
        s3.index = s3.index.rename(['bus'] + list(s3.index.names[1:]))
        s = s.add(abs(s3), fill_value = 0)
        transparency_value = None 
        bus_filtered = bus_types['buses_rural heat'].index
    if sector == 'urban decentral heat':
        s1 = network.generators_t.p.T.mul(network.snapshot_weightings.objective[0]).sum(axis=1).groupby([network.generators.bus, network.generators.carrier]).sum()
        s2 = network.storage_units_t.p.T.mul(network.snapshot_weightings.objective[0]).sum(axis=1).groupby([network.storage_units.bus, network.storage_units.carrier]).sum()
        s = s1.add(s2, fill_value = 0)
        s3 = network.links_t.p1.T.mul(network.snapshot_weightings.objective[0]).sum(axis=1).groupby([network.links.bus1, network.links.carrier]).sum()
        s3.index = s3.index.rename(['bus'] + list(s3.index.names[1:]))
        s = s.add(abs(s3), fill_value = 0)
        transparency_value = None 
        bus_filtered = bus_types['buses_urban decentral heat'].index
    if sector == 'water tanks':
        bus_filtered = (bus_types['buses_urban decentral water tanks'] + bus_types['buses_urban central water tanks'] + bus_types['buses_rural water tanks']).index 
    if sector == 'biogas':
        s1 = network.generators_t.p.T.mul(network.snapshot_weightings.objective[0]).sum(axis=1).groupby([network.generators.bus, network.generators.carrier]).sum()
        s2 = network.storage_units_t.p.T.mul(network.snapshot_weightings.objective[0]).sum(axis=1).groupby([network.storage_units.bus, network.storage_units.carrier]).sum()
        s = s1.add(s2, fill_value = 0)
        s3 = network.links_t.p1.T.mul(network.snapshot_weightings.objective[0]).sum(axis=1).groupby([network.links.bus1, network.links.carrier]).sum()
        s3.index = s3.index.rename(['bus'] + list(s3.index.names[1:]))
        s = s.add(abs(s3), fill_value = 0)
        transparency_value = None 
        bus_filtered = bus_types['buses_biogas'].index
    if sector == 'solid biomass': #this one graphs wierdly 
        s1 = network.generators_t.p.T.mul(network.snapshot_weightings.objective[0]).sum(axis=1).groupby([network.generators.bus, network.generators.carrier]).sum()
        s2 = network.storage_units_t.p.T.mul(network.snapshot_weightings.objective[0]).sum(axis=1).groupby([network.storage_units.bus, network.storage_units.carrier]).sum()
        s = s1.add(s2, fill_value = 0)
        s3 = network.links_t.p1.T.mul(network.snapshot_weightings.objective[0]).sum(axis=1).groupby([network.links.bus1, network.links.carrier]).sum()
        s3.index = s3.index.rename(['bus'] + list(s3.index.names[1:]))
        s = s.add(abs(s3), fill_value = 0)
        transparency_value = None 
        bus_filtered = bus_types['buses_solid biomass'].index
  
    s_filtered = s.loc[bus_filtered]
    #transparency_value = transparency_value.loc[bus_filtered] if 'transparency_value' in locals() else None

    #plotting the graph
    network_figure = network.plot(
        margin = 0.2, 
        ax = ax,
        bus_sizes = s_filtered / (s_filtered.max()),  
        #bus_alpha = round(((abs(transparency_value)/ s_filtered).sum() / len(bus_filtered)),2 ), 
    )

    #Adding the legend 
    
    carrier_plot_list = []
    for i in range(len(s_filtered.index)):
      carrier_plot_list.append(s_filtered.index[i][1])
    carrier_plot_list = pd.Series(carrier_plot_list)
    filter_criteria = carrier_plot_list.unique().copy() #great that looks about right 
    filter_criteria 

    #filter the tech_color dictionary to have just the carriers I want

    filtered_dict = {k: v for k, v in tech_colors.items() if k in filter_criteria}
    filtered_dict

    #turn that dictionary into the axes 
    legend_handles = [Patch(facecolor=color, label=carrier) for carrier, color in filtered_dict.items()]
    # Add the legend
    ax.legend(
        handles=legend_handles,
        title="Carriers",
        loc="center right",
        frameon=False,
        bbox_to_anchor=(0, 0.5),
        ncols = 1,
        prop={'size': 18},           # <-- Change this value to your desired font size
        title_fontsize=18 
    )

    return network_figure


#Line Statistics 
def line_statistics(network):
    #Line length
    line_length = network.lines.length.sum()
    print('The total line length as given in n.lines  is   ' + str("{:e}".format(line_length))+ '  Km')

    #Installed Line Capacity
    apparent_power_optimization = network.lines.s_nom_opt.sum()
    print('The total apparent power capacity in the optimized system is   ' + str("{:e}".format(apparent_power_optimization))  + '  MVA')
    day_of_max_power_flow = abs(network.lines_t.p0).sum(axis=1).idxmax() #is this metric even realistic? Is it the power flowing through the wire if it's taken from the bus, I mean is has to be that p0 = -p1 right
    power_max_multiplied_by_length = (abs(network.lines_t.p0).loc[day_of_max_power_flow] * network.lines.length).sum()
    print('The maximum utilization of the transmission system (i.e. when the maximum amount of power in MW is flowing through the system ) in terms of transmission size is  ' + str(power_max_multiplied_by_length) + '  MWkm')
    print()
    #Extended Line Capacity
    print('The extended capacity in the system optimization is equal to    ' + str("{:e}".format(network.statistics.expanded_capacity().Line.item())) + '  Line extended capacity  units uncertain, believe its MW by default')
    print()
    #Line Cost 
    print('The capex of the Line transmisssion system is ' + str("{:e}".format(network.statistics.capex().Line.item())) + '  unit: unsure, but I believe it is total system cost (i.e. not cost per km or MWkm) given it is billions for the whole snapshot duration (aka, for year) -- but is it annualized?') 
    print()
    print('The opex of the Line transmission system is not give, assumed 0 I suppose')




    #Line Loading on Map (to reveal potential congestion issues)
    ### I have data for every day of the year, so either need to pick a day that I want to evaluate that is random, also will probably want to pick a day when demand is largest because that will be the day the grid is most congested no? 
    ### Not necessarily if the demand is so high that local generators are turned on due to the passing of the threshold of their startup costs given the demand, will need to check if the most congested day == day of most demand

    # day in the middle of january 
    #line_loading = n.lines_t.p0.iloc[15].abs() / n.lines.s_nom / n.lines.s_max_pu * 100 
        #line loading for the day with the highest load demand
            #n.loads_t.p_set.sum(axis = 1).idxmax() #december 4 is the day with the highest load
            #n.loads_t.p_set.loc[n.loads_t.p_set.sum(axis = 1).idxmax()].sum() #highest load was 67848.44 GW 
    line_loading = network.lines_t.p0.loc[network.loads_t.p_set.sum(axis = 1).idxmax()].abs() / network.lines.s_nom / network.lines.s_max_pu * 100 
        #line loading for the most congested day
    #line_loading = n.lines_t.p0.loc[max_congestion_date].abs() / n.lines.s_nom / n.lines.s_max_pu * 100 

    norm = plt.Normalize(vmin=0, vmax=100)
    fig = plt.figure(figsize=(7, 7))
    ax = plt.axes(projection=ccrs.EqualEarth())

    network.plot(
        ax=ax,
        bus_sizes=0,
        line_colors=line_loading,
        line_norm=norm,
        line_cmap="plasma",
        line_widths=network.lines.s_nom / 2000,
        link_widths = 0,
    )

    plt.colorbar(
        plt.cm.ScalarMappable(cmap="plasma", norm=norm),
        ax=ax,
        label="Relative line loading [%]",
        shrink=0.6,
    )

    #Also, the H2 lines exist and connect buses that are not adjacent to each other, but they are not utilized at all (I wonder if runs without the hydrogen pipelines would lead to more long transmission lines being built?)
    ##n.links.p_nom_opt is used to justify this

    '''
    #loop to find day with the most congested line 
    max_congestion_date = 0 
    max_congestion_data = 0 
    for index, row in n.lines_t.p0.iterrows():
        c = index
        line_loading = n.lines_t.p0.loc[c].abs() / n.lines.s_nom / n.lines.s_max_pu * 100 
        c2 = line_loading
        if line_loading.max() > max_congestion_data:
            line_loading_max = line_loading 
            max_congestion_data = line_loading.max()
            max_congestion_date = index
'''     
    return None 



# Link Statistics 
def link_statistics(network):
    #############
    #Total European Link Statistics 
    ############
    #Link length
    link_length = network.links.length.sum()
    print('The total line length as given in n.lines  is   ' + str("{:e}".format(link_length)) + '  Km')

    #Extended Line Capacity
    print('The extended capacity in the entire European, all-sector, link optimization is equal to    ' + str("{:e}".format(network.statistics.expanded_capacity().Link.sum())) + '    units uncertain, believe its MW by default') #see https://pypsa.readthedocs.io/en/v0.29.0/api/_source/pypsa.statistics.StatisticsAccessor.expanded_capacity.html
    print()

    #Line Cost 
    print('The capex of the entire link system is ' + str("{:e}".format(network.statistics.capex().Link.sum())) + '  unit: unsure, but I believe it is total system cost (i.e. not cost per km or MWkm) given it is billions') 
    print()
    print('The opex of the entire link system is  ' + str("{:e}".format(network.statistics.opex(aggregate_time='sum').Link.sum())) + '   Euros (sum over the course of the year, so this is per annum)')


    #this is how you would plot it, but it looks a bit janky, and it's for the most conjested day of the year (.idmax())

    norm = plt.Normalize(vmin=0, vmax=100)
    fig = plt.figure(figsize=(10, 10))
    ax = plt.axes(projection=ccrs.EqualEarth())

    link_loading = network.links_t.p0.loc[network.loads_t.p_set.sum(axis = 1).idxmax()].abs() / network.links.p_nom / network.links.p_max_pu * 100 
    # day in the middle of january 
    link_loading = network.links_t.p0.iloc[15].abs() / network.links.p_nom / network.links.p_max_pu * 100 

    network.plot(
        ax=ax,
        bus_sizes=0, 
        line_widths= 0,
        link_colors = link_loading,
        link_norm = norm, 
        link_cmap= 'plasma',
        link_widths= network.links.p_nom / 1e3
    )

    plt.colorbar(
        plt.cm.ScalarMappable(cmap="plasma", norm=norm),
        ax=ax,
        label="Relative link loading [%]",
        shrink=0.6,
    )

    #maybe really think one is blue because it is SO large that it is just  not often at full capacity

    return None

