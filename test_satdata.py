import pickle


with open("sat-data/visibles_t_u-46users-filtered.pkl", "rb") as f:
    visible_t_u = pickle.load(f)

print(len(visible_t_u))
print(len(visible_t_u[0]))
print(len(visible_t_u[0][0]))
print(visible_t_u[0][0].keys())
print(type(visible_t_u[0][0][1143]))
print(visible_t_u[0][0][1143].range)