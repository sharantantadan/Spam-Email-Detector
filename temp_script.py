# Read the file
file_path = r'c:\Users\kband\OneDrive\Desktop\spam email\spam_detector.py'

with open(file_path, 'r') as f:
    lines = f.readlines()

# Find the index of the line with "if __name__"
if_name_index = None
for i, line in enumerate(lines):
    if 'if __name__' in line:
        if_name_index = i
        break

if if_name_index is not None:
    # Create the email-check endpoint code
    email_check_endpoint = '''
@app.route("/email-check", methods=["POST"])
def email_check():
    data = request.get_json()
    email = data.get("email", "")
    
    vec = vectorizer.transform([email])
    pred = model.predict(vec)[0]
    
    return jsonify({
        "email": email,
        "is_spam": pred == 1,
        "result": "SPAM" if pred == 1 else "HAM"
    })

'''
    
    # Insert the new endpoint before the if __name__ block
    lines.insert(if_name_index, email_check_endpoint)
    
    # Write back to the file
    with open(file_path, 'w') as f:
        f.writelines(lines)
    
    print("Success! Email-check endpoint added before if __name__ block")
else:
    print("Error: Could not find 'if __name__' line")
