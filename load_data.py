import csv


def table_data_filter(data_path, name_eng=None, name_ch=None, date=None, star=None, comment=None, like=None):
    """
        从表文件中根据给出的条件过滤符合要求的数据
        input：数据路径，要查询的条件
        return：符合当前条件下的所有数据的list
    """

    with open(data_path, newline='', encoding='utf-8') as csv_file:

        return_list = []

        spam_reader = csv.reader(csv_file)

        for i, row in enumerate(spam_reader):

            match = True
            return_dict = {}

            this_name_eng = row[1]
            this_name_ch = row[2]
            this_date = row[6]
            this_star = row[7]
            this_comment = row[8]
            this_like = row[9]

            if match and name_eng != None and this_name_eng != name_eng:
                match = False
            else:
                return_dict["name_eng"] = this_name_eng

            if match and name_ch != None and this_name_ch != name_ch:
                match = False
            else:
                return_dict["name_ch"] = this_name_ch

            if match and date != None and this_date != date:
                match = False
            else:
                return_dict["date"] = this_date

            if match and star != None and this_star != star:
                match = False
            else:
                return_dict["star"] = this_star

            if match and comment != None and this_comment != comment:
                match = False
            else:
                return_dict["comment"] = this_comment

            if match and like != None and this_like != like:
                match = False
            else:
                return_dict["like"] = this_like

            if match:
                return_list.append(return_dict)

        return return_list
