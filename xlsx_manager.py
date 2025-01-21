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
    try:
        workbook = openpyxl.load_workbook(excel_filename)
        worksheet = workbook.active
        
        header_row = [cell.value for cell in worksheet[1]]
        column_index = header_row.index(column_name) if column_name in header_row else None
        
        column_data = []
        if column_index is not None:
            for row in worksheet.iter_rows(min_row=2, values_only=True):
                column_data.append(row[column_index])
                
        if len(column_data) == 0:
            raise ValueError(f"Key \"{column_name}\" not found in the Excel file.")
        
        return column_data
    
    except FileNotFoundError:
        raise FileNotFoundError(f"Excel file \"{excel_filename}\" not found. You are likely to read from a non-existent file.")
