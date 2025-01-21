import openpyxl

def store_to_excel(data_list, excel_filename):
    if not data_list:
        return
    
    workbook = openpyxl.Workbook()
    worksheet = workbook.active
    
    # Write headers
    column_headers = list(data_list[0].keys())
    worksheet.append(column_headers)
    
    # Write data rows
    for data_row in data_list:
        worksheet.append([data_row[header] for header in column_headers])
    
    workbook.save(excel_filename)

def read_from_excel(excel_filename, column_name="query"):
    column_data = []
    workbook = openpyxl.load_workbook(excel_filename)
    worksheet = workbook.active
    
    header_row = [cell.value for cell in worksheet[1]]
    column_index = header_row.index(column_name) if column_name in header_row else None
    
    if column_index is not None:
        for row in worksheet.iter_rows(min_row=2, values_only=True):
            column_data.append(row[column_index])
    
    return column_data
