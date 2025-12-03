import xml.etree.ElementTree as ET

def find_production_pathways(chain_file, target_nuclide):
    """
    Find all reactions and decays that produce a target nuclide.
    
    Parameters:
    -----------
    chain_file : str
        Path to the depletion chain XML file
    target_nuclide : str
        Target nuclide name (e.g., 'U238', 'Pu239')
    
    Returns:
    --------
    dict : Dictionary with 'reactions' and 'decays' lists
    """
    tree = ET.parse(chain_file)
    root = tree.getroot()
    
    results = {
        'reactions': [],
        'decays': []
    }
    
    # Iterate through all nuclides
    for nuclide in root.findall('.//nuclide'):
        parent_name = nuclide.get('name')
        half_life = nuclide.get('half_life', 'stable')
        decay_constant = nuclide.get('decay_constant', '0')
        
        # Check reactions
        for reaction in nuclide.findall('.//reaction'):
            reaction_type = reaction.get('type')
            target_attr = reaction.get('target')
            Q_value = reaction.get('Q', 'N/A')
            branching_ratio = reaction.get('branching_ratio', '1.0')
            
            # Some chains use 'target', others use child elements
            if target_attr == target_nuclide:
                results['reactions'].append({
                    'parent': parent_name,
                    'type': reaction_type,
                    'target': target_nuclide,
                    'Q_value': Q_value,
                    'branching_ratio': branching_ratio
                })
            
            # Check for target in child elements
            for product in reaction.findall('.//product'):
                if product.text == target_nuclide:
                    results['reactions'].append({
                        'parent': parent_name,
                        'type': reaction_type,
                        'target': target_nuclide,
                        'Q_value': Q_value,
                        'branching_ratio': branching_ratio
                    })
        
        # Check decay modes
        for decay in nuclide.findall('.//decay'):
            decay_type = decay.get('type')
            target_attr = decay.get('target')
            branching_ratio = decay.get('branching_ratio', '1.0')
            
            if target_attr == target_nuclide:
                results['decays'].append({
                    'parent': parent_name,
                    'type': decay_type,
                    'target': target_nuclide,
                    'half_life': half_life,
                    'decay_constant': decay_constant,
                    'branching_ratio': branching_ratio
                })
            
            # Check for target in child elements
            for product in decay.findall('.//product'):
                if product.text == target_nuclide:
                    results['decays'].append({
                        'parent': parent_name,
                        'type': decay_type,
                        'target': target_nuclide,
                        'half_life': half_life,
                        'decay_constant': decay_constant,
                        'branching_ratio': branching_ratio
                    })
    
    return results

# Usage
chain_file = 'data\chain_casl_pwr.xml'
target = 'U238'

pathways = find_production_pathways(chain_file, target)

print(f"\n=== Production Pathways for {target} ===\n")

if pathways['reactions']:
    print("REACTIONS:")
    for rxn in pathways['reactions']:
        print(f"  {rxn['parent']} + n → {rxn['target']} (via {rxn['type']})")
        print(f"    Q-value: {rxn['Q_value']} MeV")
        print(f"    Branching ratio: {rxn['branching_ratio']}")
        print(f"    Note: Cross section is flux-dependent, stored in ENDF/B library")
        print()
else:
    print("REACTIONS: None found")

if pathways['decays']:
    print("\nDECAYS:")
    for decay in pathways['decays']:
        print(f"  {decay['parent']} → {decay['target']} (via {decay['type']})")
        print(f"    Half-life: {decay['half_life']} s")
        print(f"    Decay constant: {decay['decay_constant']} s⁻¹")
        print(f"    Branching ratio: {decay['branching_ratio']}")
        print()
else:
    print("\nDECAYS: None found")

print(f"\nTotal pathways found: {len(pathways['reactions']) + len(pathways['decays'])}")