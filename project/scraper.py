#!/usr/bin/env python3

#
#   Santiago Larrain
#   Natnaell Mammo
#


from time import sleep
import json
import requests
from bs4 import BeautifulSoup as bs4
import csv

def get_json_webpage(url):
    '''
    The usual stuff. But it returns a JSON
    '''
    try:
        html = requests.get(url)
        return html.json()
    except:
        # fail on any kind of error
        print ('Failed getting ', url)
        return None

def create_years_url_list():
    '''
    Creates a dictionary of the urls for getting the list of cases; one for each year'
    '''
    urls = {}
    for i in range(2015, 1954, -1):
        urls[i] = get_url_for_year(i)
    return urls

def get_cases_for_all_years(urls):
    '''
    Gets the list of urls for every case of all the years
    '''
    cases = {}
    for year in urls:
        print ('Getting ', year)
        cases[year] = get_json_webpage(urls[year])
        sleep(1)    #Don't oversaturate
    return cases

def get_url_for_year(year):

    base_url1 = 'https://api.oyez.org/cases?filter=term%3A'
    base_url2 = '&labels=true&page=0&per_page=0'
    url = base_url1+str(year)+base_url2
    return url

def get_cases_for_year(year):
    url = get_url_for_year(year)
    cases = get_cases_for_all_years({year: url})
    return cases

def save_cases(cases, filename):
    '''
    Saves downloaded cases to a JSON
    '''
    with open (filename, 'w') as f:
        json.dump(cases, f)

def load_cases(filename):
    '''
    Return prevoiusly saved cases
    '''
    with open (filename, 'r') as f:
        cases = json.load(f)
    return cases

def get_case_list(year_json):
    '''
    Given a year JSON of cases, it returns a list of urls for each
    '''
    cases_urls = {}
    for case in year_json:
        cases_urls[case['docket_number']] = case['href']
    return cases_urls

def get_individual_cases (cases_urls):
    '''
    Return the JSON of all individual case for a given dict of urls
    '''
    cases_data = {}
    largo = len(cases_urls)
    n = 0
    for docket_n in cases_urls:
        print ('Getting ', str(n), ' of ', largo)
        cases_data[docket_n] = get_json_webpage(cases_urls[docket_n])
        n +=1
        sleep(1)
    return cases_data

def do_down(download_years=False, download_cases=False):

    if download_years:
        urls = create_years_url_list()
        cases = get_cases_for_all_years(urls)
        save_cases (cases, 'cases_all_years')
        print ('Urls for all cases have been saved')
    else:
        cases = load_cases('cases_all_years')

    if download_cases:

        for year in cases:
            cases_urls = get_case_list(cases[year])
            cases_data = get_individual_cases(cases_urls)
            save_cases (cases_data, 'cases_'+str(year))

    else:

        cases_data = {}

        for year in cases:
            cases_data[year] = load_cases('cases_'+str(year))

    return cases, cases_data

def get_data(year):
    case = load_cases('cases_'+str(year))
    return case

def data_per_case (single_case):
    year = single_case['citation']['year']
    docket = single_case['docket_number']
    facts = single_case['facts_of_the_case']
    question = single_case['question']

    f = bs4(facts).text.replace('\n', '').replace('\xa0', '')
    q = bs4(question).text.replace('\n', '').replace('\xa0', '')

    return [year, docket, f, q]

def check_facts_q(case):
    if case['facts_of_the_case'] and case['question']:
        return True
    else:
        return False


def data_year(year):
    cases = get_data(year)
    rv = []
    for docket in cases:
        if check_facts_q(cases[docket]):
            rv.append(data_per_case(cases[docket]))
    return rv

def data_global():
    super_list = []
    for i in range(1955, 2016):
        super_list = super_list + data_year(i)
    return super_list

def check_year(year):
    cases = get_data(year)
    yes_facts = 0
    no_facts = 0
    for docket in cases:
        if check_facts_q(cases[docket]):
            yes_facts+=1
        else:
            no_facts+=1
    #print (yes_facts, no_facts)
    return yes_facts/(yes_facts+no_facts)

def chek_all():
    rv = {}
    for i in range(1955, 2016):
        rv[i] = check_year(i)
        #print (i, rv[i])
    return rv

def write_all():
    super_list = data_global()
    with open ('cases_55-15.csv', 'w', newline='') as f:
        csvfile = csv.writer(f, delimiter='|')
        csvfile.writerow(['year', 'docket_number', 'facts_of_the_case', 'question'])
        for line in super_list:
            csvfile.writerow(line)


if __name__ == "__main__":
    #_, _ = do_down(True, True)
    write_all()






