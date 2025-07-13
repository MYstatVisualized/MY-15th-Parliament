import csv
import glob
import matplotlib
import numpy as np
from os import path
from datetime import datetime
import matplotlib.pyplot as plt
from scipy.interpolate import make_smoothing_spline


def load_csv():
    # attributes ['name', 'party', 'ccode', 'constituency', 'gender', 'dob']
    f_mp_attr = path.join(path.dirname(__file__), '15mps.csv')
    mp_attr = csv.reader(open(f_mp_attr, 'r'), delimiter=',')
    mp_attr = list(zip(*mp_attr))

    # load data
    ddata = {}
    atte = np.zeros(len(mp_attr[3]), dtype=int)
    for f_dr in sorted(glob.glob(path.join(path.dirname(__file__), 'dataset/DR-*.csv'))):
        dr_date = f_dr.split('/')[-1].rstrip('.csv').strip('DR-')
        dr_date = dr_date[-4:] + dr_date[-6:-4] + dr_date[:2]
        reader = csv.reader(open(f_dr, 'r'), delimiter=',')
        ddata[dr_date] = list(zip(*reader))
        # check missing constituency
        for cont in mp_attr[3]:
            if cont not in ddata[dr_date][2]:
                print('Missing', cont, dr_date)
                ddata[dr_date][0] = ddata[dr_date][0] + ('Absent',)
                ddata[dr_date][1] = ddata[dr_date][1] + ('',)
                ddata[dr_date][2] = ddata[dr_date][2] + (cont,)
        # sort according to cont
        cont_sort = np.argsort(ddata[dr_date][2])
        for i in range(3):
            ddata[dr_date][i] = [ddata[dr_date][i][idx] for idx in cont_sort]
        # attendance
        atte = atte + (np.array(ddata[dr_date][0]) == 'Present').astype(int)
    atte = dict(
        # name=np.array(ddata[list(ddata.keys())[0]][1]),
        constituency=np.array(ddata[list(ddata.keys())[0]][2]),
        attendance=atte * 100 / len(ddata.keys()),
        session=np.array(sorted(list(ddata.keys())), dtype=int),
        gender=[],
        party=[],
        age=[])
    # party, age, gender
    for cont in atte['constituency']:
        mp_idx = mp_attr[3].index(cont.title())
        atte['party'].append(mp_attr[1][mp_idx])
        if mp_attr[5][mp_idx] != '':
            atte['age'].append(datetime.today().year - datetime.strptime(mp_attr[5][mp_idx].rstrip(' '), '%d %B %Y').year)
        else:
            atte['age'].append(0)
        atte['gender'].append(mp_attr[4][mp_idx])
    atte['age'] = np.array(atte['age'])
    atte['party'] = np.array(atte['party'])
    atte['gender'] = np.array(atte['gender'])
    return atte, ddata


def main():
    mps, rawdata = load_csv()
    parties = np.unique(mps['party'])
    color_party = matplotlib.colormaps['tab10'](np.linspace(0, 1, parties.shape[0]))
    session = [datetime.strptime(str(session_), '%Y%m%d').strftime('%y-%m')
               for session_ in mps['session'].tolist()]
    session = [session_ + ' (' + str(session.count(session_)) + ')'
               for idx_sess, session_ in enumerate(session) if session.index(session_) >= idx_sess]
    print(mps['session'].shape[0], 'Session:')
    print(*session, sep='\n')

    # individual mp attendance
    fig, ax = plt.subplots()
    idx_atte = np.argsort(mps['attendance'] * -1)
    attendance = mps['attendance'][idx_atte]
    constituency = mps['constituency'][idx_atte]
    party = mps['party'][idx_atte]
    mp_msk = attendance < 50
    ax.bar(constituency[mp_msk], attendance[mp_msk], label=party[mp_msk],
           color=color_party[[np.argwhere(parties == party_).flatten() for party_ in party[mp_msk]]])
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys())
    ax.set_xticklabels(ax.get_xticklabels(), rotation=25, ha='right')
    ax.set_ylabel('Attendance (%)')
    plt.title('MP with <50% Attendance')
    ax.set_ylim(0, 50)
    plt.tight_layout()
    plt.show()

    # party attendance
    fig, ax = plt.subplots()
    attedances = np.zeros(parties.shape[0])
    for pid, party in enumerate(parties):
        msk = mps['party'] == party
        attedances[pid] = mps['attendance'][msk].mean()
    idx_atte = np.argsort(attedances * -1)
    ax.bar(parties[idx_atte], attedances[idx_atte], color=color_party[idx_atte])
    ax.set_ylabel('Attendance (%)')
    plt.title('Attendance Over Party')
    ax.set_ylim(50, 100)
    plt.tight_layout()
    plt.show()

    # age attendance
    starty, endy, dy = 20, 90, 10
    years = np.array(list(range(starty, endy, dy)))
    attendances = np.zeros(int((endy - starty) / dy))
    cmap = matplotlib.colormaps['cool']
    color_age = cmap(np.linspace(0, 1, years.shape[0]))
    fig, ax = plt.subplots()
    for iy, ty in enumerate(years):
        msk = (mps['age'] >= ty) * (mps['age'] < (ty + dy))
        attendances[iy] = mps['attendance'][msk].mean()
    ax.bar(years+dy/2, attendances, width=dy, edgecolor='white', color=color_age)
    ax.set_ylabel('Attendance (%)')
    ax.set_xlabel('Age')
    ax.set_ylim(50, 100)
    plt.title('Attendance Over Age Range')
    plt.tight_layout()
    plt.show()

    # attendance over time
    atten = np.zeros((mps['session'].shape[0], parties.shape[0]))
    for isess, sess in enumerate(mps['session']):
        for ipar, party in enumerate(parties):
            msk = mps['party'] == party
            atten_ = (np.array(rawdata[str(sess)][0]) == 'Present')[msk].mean()
            atten[isess, ipar] = atten_ * 100 / parties.shape[0]
    fig, ax = plt.subplots()
    x = np.arange(mps['session'].shape[0])
    # bottom = np.zeros(mps['session'].shape[0])
    # for i in range(parties.shape[0]-1, -1, -1):
    #     ax.bar(x, atten[:, i], width=1, bottom=bottom, color=color_party[i], label=parties[i])
    #     bottom += atten[:, i]
    ax.plot(x, atten.sum(axis=1), '--', color='#DB9E0040')
    a__ = make_smoothing_spline(x, atten.sum(axis=1), lam=10)(x)
    ax.plot(x, a__, color='#CB8E00FF')
    stick = np.linspace(0, mps['session'].shape[0] - 1, 8).astype(int)
    ax.set_xticks(stick, [datetime.strptime(str(tick), '%Y%m%d').strftime('%b%y') for tick in mps['session'][stick]])
    ax.set_ylabel('Attendance (%)')
    ax.set_xlabel('Parliamentary Session')
    ax.set_title('Overall Attendance Over Time')
    # handles, labels = ax.get_legend_handles_labels()
    # ax.legend(handles[::-1], labels[::-1], loc='center right')
    plt.tight_layout()
    plt.show()

    # party age composition
    starty, endy, dy = 20, 90, 14
    years = np.array(list(range(starty, endy, dy)))
    cmap = matplotlib.colormaps['cool']
    color_age = cmap(np.linspace(0, 1, years.shape[0]))
    party_age = []
    for party_ in parties:
        party_age_ = []
        for iy, ty in enumerate(years):
            msk = (mps['party'] == party_) * (mps['age'] >= ty) * (mps['age'] < (ty + dy))
            party_age_.append(msk.sum())
        party_age.append(party_age_)
    party_age = np.array(party_age)
    fig, ax = plt.subplots(2, int(np.ceil(parties.shape[0] / 2.)))
    for i in range(parties.shape[0]):
        # pie chart
        ij = i // ax.shape[1], i % ax.shape[1]
        wedges, texts, _ = ax[ij[0], ij[1]].pie(party_age[i], radius=1.4, colors=color_age,
                                                startangle=-90, counterclock=False,
                                                wedgeprops=dict(width=0.9),
                                                textprops=dict(size='larger'), pctdistance=0.75,
                                                autopct=lambda pct: str(int(np.round(pct / 100. * party_age[i].sum().item()))) if pct > 5 else '')
        ax[ij[0], ij[1]].set_title(parties[i], pad=10)
        # narrow pie indicator
        kw = dict(arrowprops=dict(arrowstyle="-"), zorder=0, va="center")
        for j, p in enumerate(wedges):
            ang = (p.theta2 - p.theta1) / 2. + p.theta1
            dang = np.abs(p.theta2 - p.theta1)
            if (dang == 0) or (dang > (5. / 100 * 360)):
                continue
            y = np.sin(np.deg2rad(ang))
            x = np.cos(np.deg2rad(ang))
            horizontalalignment = {-1: "right", 1: "left"}[int(np.sign(x))]
            connectionstyle = f"angle,angleA=0,angleB={ang}"
            kw["arrowprops"].update({"connectionstyle": connectionstyle})
            ax[ij[0], ij[1]].annotate(str(int(np.round(dang / 360 * party_age[i].sum().item()))),
                                      xy=(x, y), xytext=(0.5 * np.sign(x), 1.6 * y),
                                      horizontalalignment=horizontalalignment, **kw)
    # colorbar lagend
    norm = matplotlib.colors.BoundaryNorm(np.hstack((years, [years[-1] + dy])), cmap.N)
    fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap), location='bottom',
                 cax=ax[-1, -1].inset_axes([-4, -0.5, 4, 0.1]), label='Age Range')
    fig.suptitle('Age Composition of Party')
    plt.subplots_adjust(left=0.025, right=0.975, top=0.955, bottom=0.115, hspace=0, wspace=0.165)
    plt.show()

    # party gender composition
    party_gender = []
    for party_ in parties:
        mskl = (mps['gender'] == 'Lelaki') * (mps['party'] == party_)
        mskp = (mps['gender'] == 'Perempuan') * (mps['party'] == party_)
        party_gender.append([mskl.sum().item(), mskp.sum().item()])
    party_gender = np.array(party_gender)
    fig, ax = plt.subplots(2, int(np.ceil(parties.shape[0] / 2.)))
    for i in range(parties.shape[0]):
        # pie chart
        ij = i // ax.shape[1], i % ax.shape[1]
        wedges, texts, _ = ax[ij[0], ij[1]].pie(party_gender[i], radius=1.4, colors=['lightskyblue', 'plum'],
                                                startangle=-90, counterclock=False, pctdistance=0.75,
                                                wedgeprops=dict(width=0.9), textprops=dict(size='larger'),
                                                autopct=lambda pct: str(int(np.round(pct / 100. * party_gender[i].sum().item()))))
        ax[ij[0], ij[1]].set_title(parties[i], pad=10)
    fig.suptitle('Gender Composition of Party')
    plt.subplots_adjust(left=0.025, right=0.975, top=0.955, bottom=0.115, hspace=0, wspace=0.165)
    plt.show()

    # party age distribution
    starty, endy, dy = 20, 90, 14
    years = np.array(list(range(starty, endy, dy)))
    cmap = matplotlib.colormaps['cool']
    color_age = cmap(np.linspace(0, 1, years.shape[0]))
    party_age = []
    for party_ in parties:
        party_age_ = []
        for iy, ty in enumerate(years):
            msk = (mps['party'] == party_) * (mps['age'] >= ty) * (mps['age'] < (ty + dy))
            party_age_.append(msk.sum())
        party_age.append(party_age_)
    party_age = np.array(party_age)
    n_col = int(np.ceil(parties.shape[0] / 2.))
    for i in range(parties.shape[0]):
        ax = plt.subplot(2, n_col, i + 1)
        plt.bar(years + dy / 2, party_age[i], width=dy, edgecolor='white', color=color_age)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.set_yticks([])
        ax.set_xticks(np.arange(30, 91, 30), labels=['30', '60', '90'])
        plt.title(parties[i], pad=-5)

    plt.suptitle('Age Distribution of Party')
    plt.tight_layout()
    plt.show()

    # parliament age distribution
    starty, endy, dy = 20, 90, 10
    years = np.array(list(range(starty, endy, dy)))
    print(mps['age'].mean(), np.median(mps['age']))
    party_age = []
    for party_ in parties:
        party_age_ = []
        for iy, ty in enumerate(years):
            msk = (mps['party'] == party_) * (mps['age'] > ty) * (mps['age'] <= (ty + dy))
            party_age_.append(msk.sum())
        party_age.append(party_age_)
    party_age = np.array(party_age)
    fig, ax = plt.subplots()
    bottom = np.zeros(party_age.shape[1])
    for i in range(parties.shape[0]-1, -1, -1):
        ax.bar(years + dy / 2, party_age[i], width=dy, bottom=bottom,
               color=color_party[i], label=parties[i])
        bottom += party_age[i]
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[::-1], labels[::-1])
    plt.suptitle('Age Distribution of Parliament')
    plt.xlabel('Age')
    plt.ylabel('Head Count')
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
