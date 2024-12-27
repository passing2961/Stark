# | symbol -> 나중에 제거해도됨 (깃헙 배포할때)
OLD_PERSONA_CATEGORY_MAP = {
    'attend_school_school_name': ['School | Name', 'school name'],
    'attend_school_school_type': ['School | Type', 'school type'],
    'employed_by_company_company': ['Employment | Company', 'company name'],
    'employed_by_general_location': ['Employment | Workplace', 'workplace'],
    'gender_gender': ['Gender', 'gender'],
    'has_age_number': ['Age', 'age'],
    'has_degree_degree_type': ['School | Degree', 'degree'],
    'has_degree_subject': ['School | Degree | Subject', 'degree subject'],
    'has_profession_profession': ['Employment | Profession', 'profession'],
    'have_children_family': ['Family Status | Children', 'children'],
    'have_family_family': ['Family Status', 'family status'],
    'have_pet_animal': ['Possession | Animal', 'animal'],
    'have_sibling_family': ['Family Status | Sibling', 'sibling'],
    'have_vehicle_vehicle': ['Possession | Vehicle', 'vehicle'],
    'job_status_job_status': ['Employment | Job Status', 'job status'],
    'like_activity_goto': ['Preference | Location', 'location'],
    'like_activity_place': ['Preference | Place', 'place'],
    'like_activity_show': ['Preference | Show', 'show'],
    'like_activity_watching': ['Preference | Media Genre', 'media genre'],
    'like_animal': ['Preference | Animal', 'animal'],
    'like_book_author': ['Preference | Book | Author', 'book author'],
    'like_book_genre': ['Preference | Book | Genre', 'book genre'],
    'like_book_title': ['Preference | Book | Title', 'book title'],
    'like_color': ['Preference | Color', 'color'],
    'like_drink': ['Preference | Drink', 'drink'],
    'like_food': ['Preference | Food', 'food'],
    'like_hobby': ['Preference | Hobby', 'hobby'],
    'like_movie_genre': ['Preference | Movie | Genre', 'movie genre'],
    'like_movie_title': ['Preference | Movie | Title', 'movie_title'],
    'like_music_genre': ['Preference | Music | Genre', 'music genre'],
    'like_music_instrument': ['Preference | Music | Instrument', 'music instrument'],
    'like_music_artist': ['Preference | Music | Artist', 'music artist'],
    'like_season': ['Preference | Season', 'season'],
    'like_sport': ['Preference | Sport', 'sport'],
    'live_in_citystatecountry_citystate': ['Location | Residence', 'city-state'],
    'live_in_citystatecountry_country': ['Location | Residence', 'country'],
    'marital_status_marital': ['Marital Status', 'marital status'],
    'misc_attribute_personality_trait': ['Personal Characteristic | Personality Trait', 'personality trait'],
    'nationality_country': ['Location | Nationality', 'nationality'],
    'other_person_label': ['Personal Characteristic | Eating Habit', 'eating habit'],
    'physical_attribute_person_attribute': ['Personal Characteristic | Physical Attribute', 'physical attribute'],
    'place_origin_citystate': ['Location | Birthplace', 'city-state'],
    'place_origin_country': ['Location | Birthplace', 'country'],
    'previous_profession_profession': ['Employment | Previous Profession', 'profession'],
    'teach_subject': ['Employment | Teaching Experience | Subject', 'subject'],
    'teach_activity': ['Employment | Teaching Experience | Activity', 'activity'],
    'school_status_school_status': ['School | Status', 'school status'],
}


ALL_PERSONA_CATEGORY_MAP = {
    'attend_school_school_name': ['School ⊃ Name', 'school name'],
    'attend_school_school_type': ['School ⊃ Type', 'school type'],
    'employed_by_company_company': ['Employment ⊃ Company', 'company name'],
    'employed_by_general_location': ['Employment ⊃ Workplace', 'workplace'],
    'gender_gender': ['Gender', 'gender'],
    'has_age_number': ['Age', 'age'],
    'has_degree_degree_type': ['School ⊃ Degree', 'degree'],
    'has_degree_subject': ['School ⊃ Degree Subject', 'degree subject'],
    'has_profession_profession': ['Employment ⊃ Profession', 'profession'],
    'have_children_family': ['Family Status, Children', 'children'],
    'have_family_family': ['Family Status', 'family status'],
    'have_pet_animal': ['Possession ⊃ Animal', 'animal'],
    'have_sibling_family': ['Family Status ⊃ Sibling', 'sibling'],
    'have_vehicle_vehicle': ['Possession ⊃ Vehicle', 'vehicle'],
    'job_status_job_status': ['Employment ⊃ Job Status', 'job status'],
    'like_activity_goto': ['Preference ⊃ Location', 'location'],
    'like_activity_place': ['Preference ⊃ Place', 'place'],
    'like_activity_show': ['Preference ⊃ Show', 'show'],
    'like_activity_watching': ['Preference ⊃ Media Genre', 'media genre'],
    'like_animal': ['Preference ⊃ Animal', 'animal'],
    'like_book_author': ['Preference ⊃ Book Author', 'book author'],
    'like_book_genre': ['Preference ⊃ Book Genre', 'book genre'],
    'like_book_title': ['Preference ⊃ Book Title', 'book title'],
    'like_color': ['Preference ⊃ Color', 'color'],
    'like_drink': ['Preference ⊃ Drink', 'drink'],
    'like_food': ['Preference ⊃ Food', 'food'],
    'like_hobby': ['Preference ⊃ Hobby', 'hobby'],
    'like_movie_genre': ['Preference ⊃ Movie Genre', 'movie genre'],
    'like_movie_title': ['Preference ⊃ Movie Title', 'movie_title'],
    'like_music_genre': ['Preference ⊃ Music Genre', 'music genre'],
    'like_music_instrument': ['Preference ⊃ Music Instrument', 'music instrument'],
    'like_music_artist': ['Preference ⊃ Music Artist', 'music artist'],
    'like_season': ['Preference ⊃ Season', 'season'],
    'like_sport': ['Preference ⊃ Sport', 'sport'],
    'live_in_citystatecountry_citystate': ['Location ⊃ Residence', 'city-state'],
    'live_in_citystatecountry_country': ['Location ⊃ Residence', 'country'],
    'marital_status_marital': ['Marital Status', 'marital status'],
    'misc_attribute_personality_trait': ['Personal Characteristic ⊃ Personality Trait', 'personality trait'],
    'nationality_country': ['Location ⊃ Nationality', 'nationality'],
    'other_person_label': ['Personal Characteristic ⊃ Eating Habit', 'eating habit'],
    'physical_attribute_person_attribute': ['Personal Characteristic ⊃ Physical Attribute', 'physical attribute'],
    'place_origin_citystate': ['Location ⊃ Birthplace', 'city-state'],
    'place_origin_country': ['Location ⊃ Birthplace', 'country'],
    'previous_profession_profession': ['Employment ⊃ Previous Profession', 'profession'],
    'teach_subject': ['Employment ⊃ Teaching Experience ⊃ Subject', 'subject'],
    'teach_activity': ['Employment ⊃ Teaching Experience ⊃ Activity', 'activity'],
    'school_status_school_status': ['School ⊃ Status', 'school status'],
}


PERSONA_CATEGORY_MAP = {
    'like_activity_goto': ['Preference ⊃ Location', 'location'],
    'like_activity_place': ['Preference ⊃ Place', 'place'],
    'like_activity_show': ['Preference ⊃ Show', 'show'],
    'like_activity_watching': ['Preference ⊃ Media Genre', 'media genre'],
    'like_animal': ['Preference ⊃ Animal', 'animal'],
    'like_book_author': ['Preference ⊃ Book Author', 'book author'],
    'like_book_genre': ['Preference ⊃ Book Genre', 'book genre'],
    'like_book_title': ['Preference ⊃ Book Title', 'book title'],
    'like_color': ['Preference ⊃ Color', 'color'],
    'like_drink': ['Preference ⊃ Drink', 'drink'],
    'like_food': ['Preference ⊃ Food', 'food'],
    'like_hobby': ['Preference ⊃ Hobby', 'hobby'],
    'like_movie_genre': ['Preference ⊃ Movie Genre', 'movie genre'],
    'like_movie_title': ['Preference ⊃ Movie Title', 'movie_title'],
    'like_music_genre': ['Preference ⊃ Music Genre', 'music genre'],
    'like_music_instrument': ['Preference ⊃ Music Instrument', 'music instrument'],
    'like_music_artist': ['Preference ⊃ Music Artist', 'music artist'],
    'like_season': ['Preference ⊃ Season', 'season'],
    'like_sport': ['Preference ⊃ Sport', 'sport'],
    'misc_attribute_personality_trait': ['Personal Characteristic ⊃ Personality Trait', 'personality trait'],
    'other_person_label': ['Personal Characteristic ⊃ Eating Habit', 'eating habit'],
    'physical_attribute_person_attribute': ['Personal Characteristic ⊃ Physical Attribute', 'physical attribute'],
}

PERSONA_CATEGORY_LIST = [
    ['School ⊃ Name', 'school name'], 
    ['School ⊃ Type', 'school type'], 
    ['Employment ⊃ Company', 'company name'], 
    ['Employment ⊃ Workplace', 'workplace'], 
    ['Gender', 'gender'], ['Age', 'age'], 
    ['School ⊃ Degree', 'degree'], ['School ⊃ Degree Subject', 'degree subject'], 
    ['Employment ⊃ Profession', 'profession'], 
    ['Family Status, Children', 'children'], 
    ['Family Status', 'family status'], 
    ['Possession ⊃ Animal', 'animal'], 
    ['Family Status ⊃ Sibling', 'sibling'], 
    ['Possession ⊃ Vehicle', 'vehicle'], ['Employment ⊃ Job Status', 'job status'], 
    ['Preference ⊃ Location', 'location'], ['Preference ⊃ Place', 'place'], ['Preference ⊃ Show', 'show'], 
    ['Preference ⊃ Media Genre', 'media genre'], ['Preference ⊃ Animal', 'animal'], ['Preference ⊃ Book Author', 'book author'], 
    ['Preference ⊃ Book Genre', 'book genre'], ['Preference ⊃ Book Title', 'book title'], ['Preference ⊃ Color', 'color'], 
    ['Preference ⊃ Drink', 'drink'], ['Preference ⊃ Food', 'food'], ['Preference ⊃ Hobby', 'hobby'], ['Preference ⊃ Movie Genre', 'movie genre'], 
    ['Preference ⊃ Movie Title', 'movie_title'], ['Preference ⊃ Music Genre', 'music genre'], ['Preference ⊃ Music Instrument', 'music instrument'], 
    ['Preference ⊃ Music Artist', 'music artist'], ['Preference ⊃ Season', 'season'], ['Preference ⊃ Sport', 'sport'], 
    ['Location ⊃ Residence', 'city-state'], ['Location ⊃ Residence', 'country'], ['Marital Status', 'marital status'], 
    ['Personal Characteristic ⊃ Personality Trait', 'personality trait'], ['Location ⊃ Nationality', 'nationality'], 
    ['Personal Characteristic ⊃ Eating Habit', 'eating habit'], ['Personal Characteristic ⊃ Physical Attribute', 'physical attribute'], 
    ['Location ⊃ Birthplace', 'city-state'], ['Location ⊃ Birthplace', 'country'], ['Employment ⊃ Previous Profession', 'profession'], 
    ['Employment ⊃ Teaching Experience ⊃ Subject', 'subject'], ['Employment ⊃ Teaching Experience ⊃ Activity', 'activity'], ['School ⊃ Status', 'school status'],
    ['Physical Symptom', 'physical symptom'],
    ['Psychiatric Symptom', 'psychiatric symptom'],
    ['Respiratory Disease', 'respiratory disease'],
    ['Digestive Disease', 'digestive disease'],
    ['Medicine', 'medicine'],
    ['Preference ⊃ Game', 'game'], ['Preference ⊃ Fashion', 'fashion'], ['Preference ⊃ Social Media', 'social media'],
    ['Preference ⊃ Health & Fitness', 'health & fitness'], ['Preference ⊃ Technology', 'technology'], ['Preference ⊃ Art & Design', 'art & design'],
    ['Preference ⊃ Travel', 'travel'], ['Preference ⊃ Politic', 'politic'], ['Preference ⊃ Social Issue', 'social issue'],
    ['Preference ⊃ Science', 'science']
]

EXCLUDE_COMMONSENSE_TARGET = [
    'Gender', 'Age', 'Family Status, Children', 'Family Status', 'Family Status ⊃ Sibling', 'Marital Status',
    'Personal Characteristic ⊃ Personality Trait', 'Location ⊃ Nationality', 'Personal Characteristic ⊃ Eating Habit', 'Personal Characteristic ⊃ Physical Attribute',
    'Location ⊃ Birthplace', # 'Location ⊃ Residence', 
]

DEMOGRAPHIC_TARGET = [
    'Gender', 'Age', 'Marital Status',
    ['Employment ⊃ Company', 'Employment ⊃ Profession', 'Employment ⊃ Workplace'],
    'Location ⊃ Residence'
]

COMMONSENSE_TARGET = [
    'Possession ⊃ Animal', 'Possession ⊃ Vehicle', 
    'Preference ⊃ Location', 'Preference ⊃ Place', 'Preference ⊃ Show', 'Preference ⊃ Media Genre',
    'Preference ⊃ Animal', 'Preference ⊃ Book Author', 'Preference ⊃ Book Genre', 'Preference ⊃ Book Title',
    'Preference ⊃ Color', 'Preference ⊃ Drink', 'Preference ⊃ Food', 'Preference ⊃ Hobby',
    'Preference ⊃ Movie Genre', 'Preference ⊃ Movie Title', 'Preference ⊃ Music Genre',
    'Preference ⊃ Music Instrument', 'Preference ⊃ Music Artist', 'Preference ⊃ Season',
    'Preference ⊃ Sport', 'Employment ⊃ Company', 'Employment ⊃ Profession', 'Employment ⊃ Workplace'
]

PEACOK_RELATION = [
    'characteristic', 'experience', 'goal', 'relationship', 'routine'
]

AGE_LIST = [
    '10-20', '20-30', '30-40', '40-50', '50-60', '60-70', '70-80', '80-90'
]

GENDER_LIST = [
    'Male', 'Female', #'Non-binary'
]

EDUCATION_LIST = [
    'Colleague',
    'Graduate School',
    'High School',
    'Ph.D.',
    'Pre-High School',
    'Professional School',
    'Elementary School',
    'Middle School',
    'Master’s Degree'
]

COUNTRY_LIST = [
   'United States of America', 'China', 'Japan', 'India', 'United Arab Emirates',
   'France', 'Germany', 'Italy', 'South Korea', 'Saudi Arabia', 
   'Kazakhstan', 'Brazil',
   'Mexico', 'Egypt', 'Argentina', 'Russia', 'United Kingdom', 
   'Spain', 'Canada'
]



# 'Thailand', 

COUNTRY_NAME2ALPHA = {
    'United States of America': 'US', 
    'China': 'CN', 
    'Japan': 'JP', 
    'India': 'IN', 
    'United Arab Emirates': 'AE',
    'France': 'FR', 
    'Germany': 'DE', 
    'Italy': 'IT', 
    'South Korea': 'KR', 
    'Saudi Arabia': 'SA', 
    'Kazakhstan': 'KZ', 
    'Brazil': 'BR',
    'Mexico': 'MX', 
    'Egypt': 'EG', 
    'Argentina': 'AR', 
    'Russia': 'RU', 
    'United Kingdom': 'GB', 
    'Spain': 'ES', 
    'Canada': 'CA'
}

COUNTRY_ALPHA_LIST = [
    'US', # United States # United States of America
    'CN', # China # People's Republic of China
    'JP', # Japan # Japan
    'IN', # India # Republic of India
    'AE', # United Arab Emirates # United Arab Emirates
    'FR', # France # French Republic
    'DE', # Germany # Federal Republic of Germany
    'IT', # Italy # Italian Republic
    'KR', # Korea, Republic of # Korea, Republic of
    'SA', # Saudi Arabia # Kingdom of Saudi Arabia
    'KZ', # Kazakhstan # Republic of Kazakhstan
    'BR', # Brazil # Federative Republic of Brazil
    'MX', # Mexico # United Mexican States
    'EG', # Egypt # Arab Republic of Egypt
    'AR', # Argentina # Argentine Republic
    'RU', # Russian Federation # Russian Federation
    'GB', # United Kingdom # United Kingdom of Great Britain and Northern Ireland
    'ES', # Spain # Kingdom of Spain
    'CA', # Canada # Canada
]



ETHNICITY_LIST = [
    'Asian',
    'South Asian',
    'Southeast Asian',
    'East Asian',
    'Black / African American',
    'Latino / Latinx / Hispanic',
    'Native American / Alaskan Native / Indigenous American / First Nations / American Indian',
    'Native Australian',
    'European',
    'Sub-Saharan African',
    'Middle Eastern',
    'Native Hawaiian / Pacific Islander',
    'Multiracial',
    'White / Caucasian',
    'American'
]

RELIGION_LIST = [
    'Buddhist', 'Christian', 'Hindu', 'Jewish', 'Muslim'
]


FACE_ATTRIBUTE_CATEGORY = {
    "gender": [
        "what is the gender of the person in the image?",
        [
            "male",
            "female"
        ]
    ],
    "country": [
        "what is the country of the person in the image?",
        [
            "indian",
            "latino",
            "middle eastern",
            "african",
            "asian",
            "caucasian"
        ]
    ],
    "age": [
        "what is the age of the person in the image?",
        [
            "infant",
            "toddler",
            "child",
            "teenager",
            "adult",
            "elderly"
        ]
    ],
    "body shape": [
        "what is the body shape of the person in the image?",
        [
            "fit",
            "skinny",
            "obese",
            "muscular"
        ]
    ],
    "hair color": [
        "what is the hair color of the person?",
        [
            "Because multiple colors may be included, this question is open-ended!"
        ]
    ],
    "hair style": [
        "what is the hair style of the person?",
        [
            "wavy",
            "ponytail",
            "straight",
            "bob",
            "bald",
            "curly",
            "bun",
            "others",
            "afro-hair"
        ]
    ],
    "hair length": [
        "what is the hair length of the person?",
        [
            "below chest",
            "bald",
            "above nose",
            "above chin",
            "above shoulders",
            "above chest",
            "crew cut",
            "above eyes"
        ]
    ]
}