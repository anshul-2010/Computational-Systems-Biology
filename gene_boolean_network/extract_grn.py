import cobra

model_base = cobra.io.read_sbml_model("https://github.com/anshul-2010/Computational-Systems-Biology/blob/main/data_model/iJO1366.xml")
medium = model_base.medium.copy()
test_model = model_base.copy()

model_base

## Finding the suitable genes and reactions for our analysis
rxns_itp = ["NTP10", "ATPHs", "ADK4", "NTPP9"]
genes_itp = []
genes_names_itp = []
metabolites_itp = []

# Obtain the gene reaction rules and their corresponding gene ids
for rxn in model_base.reactions:
  if rxn.id in rxns_itp:
    gene_i = rxn.gene_reaction_rule.split(" or ")
    genes_itp += gene_i

# Obtain the gene names corresponding to the gene ids
for gene in model_base.genes:
  if str(gene) in genes_itp:
    genes_names_itp.append(gene.name)

# Obtain the rxn metabolites ids
for rxn in model_base.reactions:
  if rxn.id in rxns_itp:
    metabolites = [meta.id for meta in rxn.metabolites]
    metabolites_itp += metabolites

metabolites_itp = list(set(metabolites_itp))

print(rxns_itp)
print(genes_itp)
print(genes_names_itp)
print(metabolites_itp)