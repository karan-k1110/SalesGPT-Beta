import boto3
from langchain.agents import Tool
from langchain.chains import RetrievalQA
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.tools import StructuredTool
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
import datetime
from langchain_community.tools.tavily_search import TavilySearchResults
from dotenv import load_dotenv
from litellm import completion
import json
from salesgpt_beta.models import MFInput
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

def setup_knowledge_base_for_knowledge_search(
        keyword: str = None, model_name: str = "gpt-3.5-turbo"
):
    """
    We assume that the product catalog is simply a text string.
    """

    llm = ChatOpenAI(model_name="gpt-4-0125-preview", temperature=0)

    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
    docsearch = PineconeVectorStore(index_name="knowledge-center-a1", embedding=embeddings)

    knowledge_base = RetrievalQA.from_chain_type(
        llm=llm, chain_type="stuff", retriever=docsearch.as_retriever()
    )
    return knowledge_base

def setup_knowledge_base_for_product_search(
        keyword: str = None, model_name: str = "gpt-3.5-turbo"
):
    """
    We assume that the product catalog is simply a text string.
    """

    llm = ChatOpenAI(model_name="gpt-4-0125-preview", temperature=0)

    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
    docsearch = PineconeVectorStore(index_name="product-catalog", embedding=embeddings)

    knowledge_base = RetrievalQA.from_chain_type(
        llm=llm, chain_type="stuff", retriever=docsearch.as_retriever()
    )
    return knowledge_base


def setup_knowledge_base_for_risk_factors(
        keyword: str = None, model_name: str = "gpt-3.5-turbo"
):
    """
    We assume that the product catalog is simply a text string.
    """

    llm = ChatOpenAI(model_name="gpt-4-0125-preview", temperature=0)

    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
    docsearch = PineconeVectorStore(index_name="risk-factors", embedding=embeddings)

    knowledge_base = RetrievalQA.from_chain_type(
        llm=llm, chain_type="stuff", retriever=docsearch.as_retriever()
    )
    return knowledge_base


def setup_knowledge_base_for_charges(
        keyword: str = None, model_name: str = "gpt-3.5-turbo"
):
    """
    We assume that the product catalog is simply a text string.
    """

    llm = ChatOpenAI(model_name="gpt-4-0125-preview", temperature=0)

    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
    docsearch = PineconeVectorStore(index_name="charges", embedding=embeddings)

    knowledge_base = RetrievalQA.from_chain_type(
        llm=llm, chain_type="stuff", retriever=docsearch.as_retriever()
    )
    return knowledge_base


def get_current_timestamp(query: str):
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def search_internet(query: str):
    search = TavilySearchResults()
    res = search.search(query)
    return res


def generate_calendly_link(query: str):
    return f"""Open this URL in your browser to schedule an appointment or meeting -> 
    https://calendly.com/karan111098/discussing-your-investment-needs?month=2024-08"""

def get_investment_values(query: str):
    return f"""Whats your preferred Monthly SIP amount that you are comfortable to invest? Along with yearly interest 
    and your total investment period!"""

def get_yearly_interest_rate(query: str):
    return f"""The yearly interest_rate string query is {query} """

def get_investment_period(query: str):
    return f"""The total yearly period string query is {query} """

def calculate_mf_return(monthly_amount: int, interest_rate: float, period: int):
    monthly_rate = interest_rate / 12 / 100
    months = period * 12
    future_value = monthly_amount * ((((1 + monthly_rate) ** (months)) - 1) * (1 + monthly_rate)) / monthly_rate
    future_value = round(future_value)
    total_investment = months * monthly_amount
    return f""" The total value of your investment at {interest_rate}% pa after {period} Years will be {future_value}, with total invested amount as {total_investment} and estimated returns as {future_value - total_investment}"""


def extract_values_for_sip_calculator(query: str):
    prompt = f"""
        Given the query: "{query}", analyze the content and extract the necessary information to to calculate estimated return using a SIP calculator. The information needed includes the monthly sip amount in rupees (should not include any abreviation like K or L)
        , interest rate, total investment period (in years). 
        Based on the analysis, return a dictionary in Python format where the keys are 'monthly_amount', 'interest_rate', and 'period', and the values are the corresponding pieces of information extracted from the query. 
        For example, if the query was about calculating estimated return based on investment, the output should look like this:
        {{
            "monthly_amount": "5000",
            "interest_rate": "12",
            "period": "5"
        }}
        If the input does not have all the values required, assign NaN to that specific value. 
        Don't fill in any assumed values! strictly follow the user query only!
        Now, based on the provided query, return the structured information as described.
        Return a valid directly parsable json, dont return in it within a code snippet or add any kind of explanation!!
        """
    model_name = "gpt-3.5-turbo-1106"
    response = completion(
        model=model_name,
        messages=[{"content": prompt, "role": "user"}],
        max_tokens=1000,
        temperature=0.2,
    )
    return response.choices[0].message.content.strip()

def mf_sip_calculator(query: str):
    # sip_details = {
    #         "monthly_amount": "5000",
    #         "interest_rate": "12",
    #         "period": "5"
    #     }
    sip_details = extract_values_for_sip_calculator(query)
    print(sip_details)
    if isinstance(sip_details, str):
        sip_details = json.loads(sip_details)

    print(sip_details)
    if not sip_details["monthly_amount"]:
        return f""" Take input from User for Monthly SIP Amount"""

    if not sip_details["interest_rate"]:
        return f""" Take input from User for Interest rate """

    if not sip_details["period"]:
        return f""" Total investment period is invalid"""

    monthly_amount = sip_details["monthly_amount"]
    if monthly_amount.isdigit():
        monthly_amount = int(monthly_amount)
    else:
        return f""" Take input from User for Monthly SIP Amount"""

    interest_rate = sip_details["interest_rate"]
    if interest_rate.isdecimal():
        interest_rate = float(interest_rate)
    else:
        return f""" Take input from User for Interest rate """

    period = sip_details["period"]
    if period.isdigit():
        period = int(period)
    else:
        return f""" Total investment period is invalid"""
    monthly_rate = interest_rate/12/100
    months = period * 12
    future_value = monthly_amount * ((((1 + monthly_rate)**(months))-1) * (1 + monthly_rate))/monthly_rate
    future_value = round(future_value)
    total_investment = months * monthly_amount
    return f""" The total value of your investment at {interest_rate}% pa after {period} Years will be {future_value}, with total invested amount as {total_investment} and estimated returns as {future_value - total_investment}"""


def get_tools():
    # we only use four tools for now, but this is highly extensible!
    knowledge_center = setup_knowledge_base_for_knowledge_search()
    risk_factors = setup_knowledge_base_for_risk_factors()
    charges = setup_knowledge_base_for_charges()
    product_search = setup_knowledge_base_for_product_search()

    tools = [
        Tool(
            name="KnowledgeCenter",
            func=knowledge_center.run,
            description="useful when you want to search about various investment and terminology like the investing "
                        "tools and options such as demat account, mutual funds, trading account, online share "
                        "trading, intraday trading, share market, IPO, Derivatives, Futures and options trading, "
                        "commodities trading, Income tax, Savings scheme, Pan card, Aadhar card, Crypto currency, "
                        "Personal finance"
        ),
        Tool(
            name="ProductSearch",
            func=product_search.run,
            description="useful when you want to search the investment products such as apps and websites provided by Angel one"
        ),
        Tool(
            name="RiskFactorsAndRecommendations",
            func=risk_factors.run,
            description="useful when you want to get information about what types of investment should different "
                        "types of investors should make, the risks involved and also the recommendations."
        ),
        Tool(
            name="Charges",
            func=charges.run,
            description="useful for when you want to fetch the charges of the platform. "
                        "Charges can only be in one of the following paired up with ""other charges"" and nothing "
                        "else-> 1. Equity, 2. Currency, 3. Commodities"
        ),
        Tool(
            name="GetCurrentDateTime",
            func=get_current_timestamp,
            description="useful for when you want to get current date time"
        ),
        Tool(
            name="SearchInternet",
            func=search_internet,
            description="useful for when you want to search any investment related information such as top IPOs in India and get information about them, only use this for IPOs and other investment related advice, and nothing else!"
        ),
        Tool(
            name="GenerateMeetingLink",
            func=generate_calendly_link,
            description="useful for when you want schedule an appointment or schedule a meeting"
        )
    ]

    return tools


def completion_bedrock():
    return None
