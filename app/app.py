from langchain.llms import AI21
import streamlit as st
from langchain import PromptTemplate, LLMChain
from snowflake.connector import connect
import re
from PIL import Image
import pandas as pd
import base64

# Execute Snowflake Query
def execute_snowflake_query(query, sf_dict):
    # Connect to Snowflake
    snowflake_conn = connect(
        user=sf_dict['user'],
        password=sf_dict['password'],
        account=sf_dict['account'],
        warehouse=sf_dict['warehouse'],
        database=sf_dict['database'],
        schema=sf_dict['schema']
    )

    # Execute the Snowflake query
    cursor = snowflake_conn.cursor()
    cursor.execute(query)
    result = cursor.fetchall()
    print(result)
    df = pd.DataFrame(result, columns=[desc[0] for desc in cursor.description])

    # Disconnect from Snowflake
    snowflake_conn.close()
    return df


################## below funcions are used to format the query#########
def format_snowflake_query(query, column_names):
    # Remove new line characters & leading/trailing whitespace

    for chars in ["`", "sql"]:
        query = query.replace("`", "")

    formatted_query = query.replace("\n", " ").strip()

    # Add a semicolon at the end of the query if missing
    if not formatted_query.endswith(';'):
        formatted_query += ';'
    try:
        # Execute Query
        query_result = execute_snowflake_query(formatted_query, sf_dict)
    except:
        # Enquote column names and table name
        formatted_query = enquote_identifiers(formatted_query, column_names)
    return formatted_query


def find_matching_columns(query, column_names):
    # Construct a regular expression pattern to match the column names
    pattern = r'\b(?:' + '|'.join(column_names) + r')\b'

    # Find all matches of the pattern in the query
    matches = re.findall(pattern, query, re.IGNORECASE)
    return list(set(matches))


def replace_word(query, word_to_replace, replacement):
    pattern = r"(?<![A-Za-z])" + re.escape(word_to_replace) + r"(?![A-Za-z])"
    replaced_query = re.sub(pattern, replacement, query)
    return replaced_query


def remove_extra_spaces(query):
    cleaned_query = re.sub(r'\s+', ' ', query)
    return cleaned_query.strip()


def enquote_identifiers(query, column_names):
    def enquote(column_name):
        if column_name[0] == ('\"') and column_name[-1] == ('\"'):
            return column_name
        else:
            return f'"{column_name}"'

    matched_columns = find_matching_columns(query, column_names)
    # Enquote column names
    for column_name in matched_columns:
        formatted_column_name = enquote(column_name)
        query = replace_word(query, column_name, formatted_column_name)

    query = remove_extra_spaces(query)
    return query


################## end funcions are used to format the query#########


def results_from_snowflake(question, sf_dict, table, columns):
    '''
    This function will take the user question and return the result from Snowflake table.
    '''
    llm_code = llm
    llm_text = llm

    # SQL Prompt
    query_template = """
    Write a Snowflake query for the below table, schema and question:\n
    - Table name {Table}\n
    - List of columns: {Columns}\n
    - Question: {question}\n

    Query should adhere to the following rules:\n
    - Column names should be case sensitive.\n
    - Don't Order the results if not necessary.\n
    - Ensure proper filtering conditions are applied to retrieve the desired subset of data.\n
    - Don't use groupby and where if not needed.\n
    - Table names to be wrapped in double quotes.\n
    - Enquote database and schema also in double quotes.\n
    - Don't add any additional keyword like copy and all if not added.\n
    - Don't add any braces unnecessarily.\n
    - Ensure groupby or where if used are applied properly.\n
    - Don't add any numbers unnecessarily.\n
    - Use proper and complete inbuild keywords.\n
    Please provide the query to retrieve the requested information:

    """

    sql_prompt = PromptTemplate(template=query_template, input_variables=["Table", "question", "Columns"])

    # Generate SQL Query
    def get_sql_query(table, question, columns):
        llm_chain = LLMChain(prompt=sql_prompt, llm=llm_code)
        response = llm_chain.run({"Table": table, "question": question, "Columns": columns})
        return response

    query = get_sql_query(table, question, columns)
    print(query)
    # Format Query
    formatted_query = format_snowflake_query(query, columns)
    print(formatted_query)

    # Execute Query
    try:
        query_result = execute_snowflake_query(formatted_query, sf_dict)
    except Exception as e:
        query_result = f"Error: {e}"

    # Generate answer from questions and query result
    def convert_sql_response_to_english(question, query_result):

        # Answer Prompts
        answer_template1 = """
        Given a question and the result of an SQL query, your task is to convert them into plain English format. You are to provide a human-readable explanation that accurately conveys the meaning of the question and the information contained in the query result. \n

        Example 1:
        Question: "What is the average total delivery cases?"
        Query Result: [("QuarterTotalDeliveryCases": 256)]

        Plain English Representation: "The average of total delivery cases is 256."

        Example 2:
        Question: "What are the total number of row counts?"
        Query Result: [("Count": "660")]

        Plain English Representation: "The total number of row counts is 660."

        Example 3:
        Question: "What is the Count of unique 'TrademarkID'?"
        Query Result: [("TrademarkID": 50)]

        Plain English Representation: "The count of unique 'TrademarkID' is 50." \n

        Question: {question}\n
        Query result: {query_result}\n\n

        Plain English Explanation:
        """

        answer_template2 = """
            Translate given question into a statement.\n\n
            Example 1-\n
            Question: What is the sales for brands last year?\n
            Answer: Sales for brands in last year is:\n\n
            Example 2-\n
            Question: What is the weekly avergae sale for this year?\n
            Answer: Weekly average sale for this year is:\n\n
            Example 3-\n
            Question: Show all unique trademark ids starting with `A` along with it's description.\n
            Answer: Below is a list of all unique trademark ids starting with `A` along with it's description:\n\n

            Question: {question}\n\n\n
            Answer:
            """
        if len(query_result) == 1:
            answer_prompt = PromptTemplate(template=answer_template1, input_variables=["query_result", "question"])
            llm_chain = LLMChain(prompt=answer_prompt, llm=llm_text)
            response = llm_chain.run({"query_result": query_result, "question": question})
        else:
            answer_prompt = PromptTemplate(template=answer_template2, input_variables=["question"])
            llm_chain = LLMChain(prompt=answer_prompt, llm=llm_text)
            response = llm_chain.run({"query_result": query_result, "question": question})
        return response

    response = convert_sql_response_to_english(question, query_result)
    return formatted_query, response, query_result


def return_schema_df(db_details_query, sf_dict):
    df = execute_snowflake_query(db_details_query, sf_dict)
    return df


# st.cache(return_schema_df)

# Define Variables

# Streamlit app
sf_dict = {'user': 'username', 'password': 'password', 'account': 'account',
           'role': 'ACCOUNTADMIN', 'warehouse': 'COMPUTE_WH', 'database': 'SNOWFLAKE_SAMPLE_DATA',
           'schema': 'TPCDS_SF10TCL'}

# database = sf_dict['database']
# schema = sf_dict['schema']

sys_path = '/Users/shadabhussain/Documents/Hackathons/SQLGenius/app'
streamlit_path = '/app/sqlgenius/app'
path = streamlit_path
# Page icon
icon = Image.open(path+'/app_images/icon.png')

# Page config
st.set_page_config(page_title="SQLGenius",
                   page_icon=icon,
                   layout="wide",
                   initial_sidebar_state="expanded"
                   )

st.title("SQLGenius")
# List of Databases
db_label = "Select Database:"
db_details_query = "SHOW SCHEMAS;"
db_df = None
llm_api_key = None
database = None
schema = None
selected_table_name= None

with open(path+"/app_images/SQLGenius.png", "rb") as f:
    data = base64.b64encode(f.read()).decode("utf-8")

    st.sidebar.markdown(
        f"""
        <div style="display:table;margin-top:-5%;margin-left:10%;margin-bottom:5%;">
            <img src="data:image/png;base64,{data}" width="240" height="135">
        </div>
        """,
        unsafe_allow_html=True,
    )

with st.sidebar:
    with st.expander("✨ Enter Credentials", True):
        llm_api_key = st.text_input("AI21 API Key", type="password")
        sf_dict['user'] = st.text_input("Snowflake Username")
        sf_dict['password'] = st.text_input("Snowflake Password", type="password")
        sf_dict['account'] = st.text_input("Snowflake Account")
        if llm_api_key and sf_dict['user'] and sf_dict['password'] and sf_dict['account']:
            llm = AI21(ai21_api_key=llm_api_key)
            db_df = return_schema_df(db_details_query, sf_dict)

if isinstance(db_df, pd.DataFrame) and llm_api_key:
    with st.sidebar:
        # Select Table
        with st.expander("✨ Select Table", True):
            # Select Database
            db_list = db_df.database_name.unique()
            database = st.selectbox(db_label, db_list)
            sf_dict['database'] = database
            if database:
                # Select Schema
                schema_label = "Select Schema:"
                schema_list = db_df[db_df.database_name == database].name.unique()
                schema = st.selectbox(schema_label, schema_list)
                sf_dict['schema'] = schema
                if schema:
                    # Select Tables or Views
                    type_label = "Select Table or View"
                    table_views = st.selectbox(type_label, ['tables', 'views'])

                    if table_views:
                        # List of Tables
                        table_label = "Select Tables:"
                        table_list_query = f"select table_name from {database}.information_schema.{table_views} where table_schema='{schema}' AND NOT (table_name LIKE ANY('%Snap%', '%Temp%', '%SNAP%'));"
                        table_list = sorted(list(execute_snowflake_query(table_list_query, sf_dict).TABLE_NAME.values))
                        # Select Table
                        selected_table_name = st.selectbox(table_label, table_list)
                        table = f'"{database}"."{schema}"."{selected_table_name}"'

if database and schema and selected_table_name:
    with st.container():
        # Fetch Column Names
        column_query = f"select ordinal_position as position, column_name, data_type,\
        case when character_maximum_length is not null \
             then character_maximum_length \
             else numeric_precision end as max_length, \
        is_nullable, \
        column_default as default_value \
        from information_schema.columns where table_schema ilike '{schema}' and table_name ilike '{selected_table_name}' order by ordinal_position;"

        column_df = execute_snowflake_query(column_query, sf_dict)
        columns = list(column_df.COLUMN_NAME.values)

        # Write Sample DF
        sample_df_query = f'select * from {table} limit 5;'
        sample_df = execute_snowflake_query(sample_df_query, sf_dict)
        st.write("Ask any question related to the table, and the AI will provide a response.")
        with st.expander("Expand for sample data:"):
            st.dataframe(sample_df)

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"], unsafe_allow_html=True)

# Accept user input
if question := st.chat_input("Type your prompt:"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": question})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(question, unsafe_allow_html=True)
    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""

if question and database and schema and selected_table_name:
    # Generate AI response
    # sample_df_query = f'select * from {table} limit 100;'
    formatted_query, response, query_result = results_from_snowflake(question, sf_dict, table,
                                                                     columns)  # ("query", "response", execute_snowflake_query(sample_df_query, sf_dict))
    if isinstance(query_result, pd.DataFrame):
        if len(query_result) == 1:
            full_response += "Query-"
            full_response += "\n" + "\n" + "```" + formatted_query + "```"
            full_response += "\n" + "\n" + "Answer-"
            full_response += "\n" + response
            message_placeholder.markdown(full_response)

            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": full_response})
        else:
            full_response += "Query-"
            full_response += "\n" + "\n" + "```" + formatted_query + "```"
            full_response += "\n" + "\n" + "Answer-"
            full_response += "\n" + response

            # Convert DataFrame to HTML table with scrollbar
            html_table = query_result.to_html(index=False)
            html_output = f'<div style="overflow-x: auto; width: 600px; height: 200px;">{html_table}</div>'
            full_response += "\n" + html_output

            # Add a download button for the CSV file
            csv = query_result.to_csv(index=False)
            b64 = base64.b64encode(csv.encode()).decode()
            filename = "data.csv"
            href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">Download CSV</a>'
            full_response += "\n" + "\n" + href

            message_placeholder.markdown(full_response, unsafe_allow_html=True)

            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": full_response})
    else:
        full_response += "Query-"
        full_response += "\n" + "\n" + "```" + formatted_query + "```"
        full_response += "\n" + "\n" + "Error Message-"
        full_response += "\n" + query_result
        message_placeholder.markdown(full_response)

        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": full_response})
