import os
import common.tool as tool
if __name__ == "__main__":
    # app的ID列表
    app_ids = []
    for i in range(1, 5):
        app_ids.append(i)
    if not os.path.exists("sat-data/apps-weights-4.csv"):
        tool.generate_app_weight_csv(app_ids, "sat-data/apps-weights-4.csv")
    sat_weights = tool.read_app_weight_csv("sat-data/apps-weights-4.csv")
    print(sat_weights)
    # satellite_ids = [1143,
    # 1644,
    # 1666,
    # 3743]
    # output_file = "sat-data/sats-bd-rb.csv"
    # tool.generate_sats_bd_rb_csv(satellite_ids, output_file)
    # sat_dict = tool.read_sats_bd_rb_csv(output_file)
    # print(sat_dict)