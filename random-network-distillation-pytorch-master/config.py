import configparser

config = configparser.ConfigParser()
config.read('./config.conf')

# ---------------------------------
default = 'DEFAULT'
grid = 'GRID'
# ---------------------------------
default_config = config[default]
grid_config = config[grid]
