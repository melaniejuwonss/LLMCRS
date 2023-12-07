import json

content_data = json.load((open('../data/content_data.json', 'r', encoding='utf-8')))[0]
movie2name = json.load((open('../data/redial/movie2name.json', 'r', encoding='utf-8')))
crsid2name = dict()
for key, value in movie2name.items():
    if key not in crsid2name.keys():
        crsid2name[key] = value[1]


def objective(content_data):
    genre_template = "%s is %s film."
    director_template = " It is directed by %s."
    writer_template = " The screenplay was written by %s."
    star_template = " The film stars %s."
    whole_template = []
    for datas in content_data:
        crs_id = datas['crs_id']
        title = crsid2name[crs_id]
        genres = ", ".join(datas['meta']['genre']).lower()
        if genres.rfind(",") != -1:
            genres = genres[:genres.rfind(",")] + ", and" + genres[genres.rfind(",") + 1:]
        directors = ", ".join(datas['meta']['director'])
        if directors.rfind(",") != -1:
            directors = directors[:directors.rfind(",")] + ", and" + directors[directors.rfind(",") + 1:]
        writers = ", ".join(datas['meta']['writers'])
        if writers.rfind(",") != 1:
            writers = writers[:writers.rfind(",")] + ", and" + writers[writers.rfind(",") + 1:]
        stars = ", ".join(datas['meta']['stars'])
        if stars.rfind(",") != -1:
            stars = stars[:stars.rfind(",")] + ", and" + stars[stars.rfind(",") + 1:]

        whole_template.append(
            {'context_tokens': (genre_template + director_template + writer_template + star_template) % (
                title, genres, directors, writers, stars), 'item': ''})

    with open('../data/passage/objective.json', 'w', encoding='utf-8') as f:
        f.write(json.dumps(whole_template, indent=2))


def plot(content_data, plot_num):
    whole_template = []
    plot_template = "The following text is a plot of movie %s.\n%s"
    for datas in content_data:
        crs_id = datas['crs_id']
        title = crsid2name[crs_id]
        summaries = datas['summary']
        for i in range(min(len(summaries), plot_num)):
            summary = summaries[i]
            whole_template.append({'context_tokens': plot_template % (title, summary), 'item': ''})
    with open(f'../data/passage/plot{plot_num}.json', 'w', encoding='utf-8') as f:
        f.write(json.dumps(whole_template, indent=2))


# objective(content_data)
plot(content_data, 1)
