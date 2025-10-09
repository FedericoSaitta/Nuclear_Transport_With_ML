import openmc.deplete
import os

# Load the decay chain
script_dir = os.path.dirname(os.path.abspath(__file__))
chain_file = "../data/chain_casl_pwr.xml"
chain = openmc.deplete.Chain.from_xml(os.path.join(script_dir, chain_file))

# List all isotopes tracked
print(f"Number of nuclides tracked before trimming: {len(chain.nuclides)}\n")

#get a chain just containing U235, U238 only
trimmed_chain = chain.reduce(['U235', 'U238'], 0) 

#prepare the results directory
results_dir = os.path.join(script_dir, "util_results", "trimmed_chain")
os.makedirs(results_dir, exist_ok=True)

#export the trimmed chain to the just prepped util_results
output_chain_file = os.path.join(results_dir, "trimmed_chain.xml")
trimmed_chain.export_to_xml(output_chain_file)

# List all isotopes kept
new_chain = openmc.deplete.Chain.from_xml(output_chain_file)
print(f"Number of nuclides tracked after trimming: {len(new_chain.nuclides)}\n")
print(f"Trimmed chain saved to: {output_chain_file}")

