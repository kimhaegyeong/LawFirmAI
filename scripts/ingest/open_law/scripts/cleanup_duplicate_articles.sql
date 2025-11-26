-- 중복 데이터 제거 SQL
-- 주의: 실행 전에 백업을 권장합니다

-- 법령ID: 9, 조문: 017600, 항: 000000, 중복 24개
-- 제목: 사전조사
DELETE FROM statutes_articles WHERE id IN (4467, 4468, 4469, 4470, 4471, 4472, 4473, 4474, 4475, 4476, 4477, 4478, 4479, 4480, 4481, 4482, 4483, 4484, 4485, 4486, 4487, 4488, 4489);

-- 법령ID: 9, 조문: 015000, 항: 000000, 중복 21개
-- 제목: 징벌의 부과기준
DELETE FROM statutes_articles WHERE id IN (4377, 4378, 4379, 4380, 4381, 4382, 4383, 4384, 4385, 4386, 4387, 4388, 4389, 4390, 4391, 4392, 4393, 4394, 4395, 4396);

-- 법령ID: 9, 조문: 014800, 항: 000000, 중복 19개
-- 제목: 규율
DELETE FROM statutes_articles WHERE id IN (4357, 4358, 4359, 4360, 4361, 4362, 4363, 4364, 4365, 4366, 4367, 4368, 4369, 4370, 4371, 4372, 4373, 4374);

-- 법령ID: 9, 조문: 009100, 항: 000000, 중복 18개
-- 제목: 기능
DELETE FROM statutes_articles WHERE id IN (4180, 4181, 4182, 4183, 4184, 4185, 4186, 4187, 4188, 4189, 4190, 4191, 4192, 4193, 4194, 4195, 4196);

-- 법령ID: 3, 조문: 002500, 항: 000000, 중복 17개
-- 제목: 민감정보 및 고유식별정보의 처리
DELETE FROM statutes_articles WHERE id IN (323, 324, 325, 326, 327, 328, 329, 330, 331, 332, 333, 334, 335, 336, 337, 338);

-- 법령ID: 9, 조문: 003100, 항: 000000, 중복 16개
-- 제목: 서신검열 등의 대상
DELETE FROM statutes_articles WHERE id IN (4003, 4004, 4005, 4006, 4007, 4008, 4009, 4010, 4011, 4012, 4013, 4014, 4015, 4016, 4017);

-- 법령ID: 16, 조문: 036100, 항: 000000, 중복 15개
-- 제목: 항소이유
DELETE FROM statutes_articles WHERE id IN (7822, 7823, 7824, 7825, 7826, 7827, 7828, 7829, 7830, 7831, 7832, 7833, 7834, 7835);

-- 법령ID: 8, 조문: 009400, 항: 000000, 중복 14개
-- 제목: 징벌의 종류
DELETE FROM statutes_articles WHERE id IN (3794, 3795, 3796, 3797, 3798, 3799, 3800, 3801, 3802, 3803, 3804, 3805, 3806);

-- 법령ID: 13, 조문: 010800, 항: 000000, 중복 14개
-- 제목: 징벌의 종류
DELETE FROM statutes_articles WHERE id IN (6207, 6208, 6209, 6210, 6211, 6212, 6213, 6214, 6215, 6216, 6217, 6218, 6219);

-- 법령ID: 16, 조문: 005100, 항: 000②00, 중복 14개
-- 제목: 공판조서의 기재요건
DELETE FROM statutes_articles WHERE id IN (6797, 6798, 6799, 6800, 6801, 6802, 6803, 6804, 6805, 6806, 6807, 6808, 6809);

-- 법령ID: 11, 조문: 000100, 항: 000④00, 중복 13개
-- 제목: 적용대상자
DELETE FROM statutes_articles WHERE id IN (4839, 4840, 4841, 4842, 4843, 4844, 4845, 4846, 4847, 4848, 4849, 4850);

-- 법령ID: 14, 조문: 014300, 항: 000①00, 중복 13개
-- 제목: 석방예정자의 수용이력 등 통보
DELETE FROM statutes_articles WHERE id IN (6623, 6624, 6625, 6626, 6627, 6628, 6629, 6630, 6631, 6632, 6633, 6634);

-- 법령ID: 16, 조문: 026600, 항: 000①00, 중복 12개
-- 제목: 공판준비에 관한 사항
DELETE FROM statutes_articles WHERE id IN (7528, 7529, 7530, 7531, 7532, 7533, 7534, 7535, 7536, 7537, 7538);

-- 법령ID: 3, 조문: 002400, 항: 000①00, 중복 11개
-- 제목: 권한의 위임
DELETE FROM statutes_articles WHERE id IN (311, 312, 313, 314, 315, 316, 317, 318, 319, 320);

-- 법령ID: 7, 조문: 045100, 항: 000①00, 중복 11개
-- 제목: 재심사유
DELETE FROM statutes_articles WHERE id IN (3356, 3357, 3358, 3359, 3360, 3361, 3362, 3363, 3364, 3365);

-- 법령ID: 9, 조문: 000200, 항: 000000, 중복 10개
-- 제목: 정의
DELETE FROM statutes_articles WHERE id IN (3893, 3894, 3895, 3896, 3897, 3898, 3899, 3900, 3901);

-- 법령ID: 9, 조문: 008800, 항: 000①00, 중복 10개
-- 제목: 귀휴 사유
DELETE FROM statutes_articles WHERE id IN (4163, 4164, 4165, 4166, 4167, 4168, 4169, 4170, 4171);

-- 법령ID: 13, 조문: 000500, 항: 000②00, 중복 10개
-- 제목: 기본계획의 수립
DELETE FROM statutes_articles WHERE id IN (5813, 5814, 5815, 5816, 5817, 5818, 5819, 5820, 5821);

-- 법령ID: 4, 조문: 004900, 항: 000②00, 중복 9개
-- 제목: 법인의 등기사항
DELETE FROM statutes_articles WHERE id IN (438, 439, 440, 441, 442, 443, 444, 445);

-- 법령ID: 4, 조문: 093700, 항: 000000, 중복 9개
-- 제목: 후견인의 결격사유
DELETE FROM statutes_articles WHERE id IN (2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012);

-- 법령ID: 9, 조문: 000300, 항: 000②00, 중복 9개
-- 제목: 직무의 처리 등
DELETE FROM statutes_articles WHERE id IN (3905, 3906, 3907, 3908, 3909, 3910, 3911, 3912);

-- 법령ID: 12, 조문: 004100, 항: 000000, 중복 9개
-- 제목: 형의 종류
DELETE FROM statutes_articles WHERE id IN (5197, 5198, 5199, 5200, 5201, 5202, 5203, 5204);

-- 법령ID: 16, 조문: 001700, 항: 000000, 중복 9개
-- 제목: 제척의 원인
DELETE FROM statutes_articles WHERE id IN (6701, 6702, 6703, 6704, 6705, 6706, 6707, 6708);

-- 법령ID: 16, 조문: 009800, 항: 000000, 중복 9개
-- 제목: 보석의 조건
DELETE FROM statutes_articles WHERE id IN (6942, 6943, 6944, 6945, 6946, 6947, 6948, 6949);

-- 법령ID: 1, 조문: 000200, 항: 000000, 중복 8개
-- 제목: 정의
DELETE FROM statutes_articles WHERE id IN (4, 5, 6, 7, 8, 9, 10);

-- 법령ID: 4, 조문: 080400, 항: 000000, 중복 8개
-- 제목: 약혼해제의 사유
DELETE FROM statutes_articles WHERE id IN (1692, 1693, 1694, 1695, 1696, 1697, 1698);

-- 법령ID: 7, 조문: 027400, 항: 000①00, 중복 8개
-- 제목: 준비서면의 기재사항
DELETE FROM statutes_articles WHERE id IN (2995, 2996, 2997, 2998, 2999, 3000, 3001);

-- 법령ID: 7, 조문: 049000, 항: 000②00, 중복 8개
-- 제목: 제권판결에 대한 불복소송
DELETE FROM statutes_articles WHERE id IN (3439, 3440, 3441, 3442, 3443, 3444, 3445);

-- 법령ID: 8, 조문: 008500, 항: 000①00, 중복 8개
-- 제목: 보호장비의 종류 및 사용요건
DELETE FROM statutes_articles WHERE id IN (3727, 3728, 3729, 3730, 3731, 3732, 3733);

-- 법령ID: 9, 조문: 012200, 항: 000000, 중복 8개
-- 제목: 보호장비의 종류
DELETE FROM statutes_articles WHERE id IN (4279, 4280, 4281, 4282, 4283, 4284, 4285);

-- 법령ID: 11, 조문: 001400, 항: 000000, 중복 8개
-- 제목: 일반이적
DELETE FROM statutes_articles WHERE id IN (4883, 4884, 4885, 4886, 4887, 4888, 4889);

-- 법령ID: 12, 조문: 005500, 항: 000①00, 중복 8개
-- 제목: 법률상의 감경
DELETE FROM statutes_articles WHERE id IN (5235, 5236, 5237, 5238, 5239, 5240, 5241);

-- 법령ID: 13, 조문: 009800, 항: 000①00, 중복 8개
-- 제목: 보호장비의 종류 및 사용요건
DELETE FROM statutes_articles WHERE id IN (6138, 6139, 6140, 6141, 6142, 6143, 6144);

-- 법령ID: 14, 조문: 014300, 항: 000②00, 중복 8개
-- 제목: 석방예정자의 수용이력 등 통보
DELETE FROM statutes_articles WHERE id IN (6636, 6637, 6638, 6639, 6640, 6641, 6642);

-- 법령ID: 3, 조문: 000500, 항: 000①00, 중복 7개
-- 제목: 출입국항에서의 난민신청자에 대한 난민인정 심사 회부
DELETE FROM statutes_articles WHERE id IN (227, 228, 229, 230, 231, 232);

-- 법령ID: 4, 조문: 004000, 항: 000000, 중복 7개
-- 제목: 사단법인의 정관
DELETE FROM statutes_articles WHERE id IN (417, 418, 419, 420, 421, 422);

-- 법령ID: 4, 조문: 009700, 항: 000000, 중복 7개
-- 제목: 벌칙
DELETE FROM statutes_articles WHERE id IN (530, 531, 532, 533, 534, 535);

-- 법령ID: 4, 조문: 016300, 항: 000000, 중복 7개
-- 제목: 3년의 단기소멸시효
DELETE FROM statutes_articles WHERE id IN (644, 645, 646, 647, 648, 649);

-- 법령ID: 8, 조문: 004400, 항: 000⑤00, 중복 7개
-- 제목: 서신수수
DELETE FROM statutes_articles WHERE id IN (3600, 3601, 3602, 3603, 3604, 3605);

-- 법령ID: 8, 조문: 008700, 항: 000①00, 중복 7개
-- 제목: 강제력의 행사
DELETE FROM statutes_articles WHERE id IN (3742, 3743, 3744, 3745, 3746, 3747);

-- 법령ID: 9, 조문: 001600, 항: 000③00, 중복 7개
-- 제목: 교부금품의 허가
DELETE FROM statutes_articles WHERE id IN (3948, 3949, 3950, 3951, 3952, 3953);

-- 법령ID: 9, 조문: 013900, 항: 000000, 중복 7개
-- 제목: 보안장비의 종류
DELETE FROM statutes_articles WHERE id IN (4321, 4322, 4323, 4324, 4325, 4326);

-- 법령ID: 10, 조문: 011300, 항: 000000, 중복 7개
-- 제목: 징벌위원회 기능
DELETE FROM statutes_articles WHERE id IN (4754, 4755, 4756, 4757, 4758, 4759);

-- 법령ID: 10, 조문: 013900, 항: 000①00, 중복 7개
-- 제목: 권한의 위임
DELETE FROM statutes_articles WHERE id IN (4821, 4822, 4823, 4824, 4825, 4826);

-- 법령ID: 11, 조문: 000200, 항: 000000, 중복 7개
-- 제목: 용어의 정의
DELETE FROM statutes_articles WHERE id IN (4854, 4855, 4856, 4857, 4858, 4859);

-- 법령ID: 12, 조문: 000500, 항: 000000, 중복 7개
-- 제목: 외국인의 국외범
DELETE FROM statutes_articles WHERE id IN (5129, 5130, 5131, 5132, 5133, 5134);

-- 법령ID: 12, 조문: 007800, 항: 000000, 중복 7개
-- 제목: 형의 시효의 기간
DELETE FROM statutes_articles WHERE id IN (5295, 5296, 5297, 5298, 5299, 5300);

-- 법령ID: 13, 조문: 004300, 항: 000⑤00, 중복 7개
-- 제목: 편지수수
DELETE FROM statutes_articles WHERE id IN (5951, 5952, 5953, 5954, 5955, 5956);

-- 법령ID: 13, 조문: 010000, 항: 000①00, 중복 7개
-- 제목: 강제력의 행사
DELETE FROM statutes_articles WHERE id IN (6153, 6154, 6155, 6156, 6157, 6158);

-- 법령ID: 16, 조문: 005900, 항: 000②00, 중복 7개
-- 제목: 재판확정기록의 열람ㆍ등사
DELETE FROM statutes_articles WHERE id IN (6833, 6834, 6835, 6836, 6837, 6838);

-- 법령ID: 16, 조문: 011400, 항: 000①00, 중복 7개
-- 제목: 영장의 방식
DELETE FROM statutes_articles WHERE id IN (6999, 7000, 7001, 7002, 7003, 7004);

-- 법령ID: 16, 조문: 024900, 항: 000①00, 중복 7개
-- 제목: 공소시효의 기간
DELETE FROM statutes_articles WHERE id IN (7431, 7432, 7433, 7434, 7435, 7436);

-- 법령ID: 16, 조문: 042000, 항: 000000, 중복 7개
-- 제목: 재심이유
DELETE FROM statutes_articles WHERE id IN (7937, 7938, 7939, 7940, 7941, 7942);

-- 법령ID: 16, 조문: 047100, 항: 000①00, 중복 7개
-- 제목: 동전
DELETE FROM statutes_articles WHERE id IN (8030, 8031, 8032, 8033, 8034, 8035);

-- 법령ID: 1, 조문: 002200, 항: 000②00, 중복 6개
-- 제목: 난민인정결정의 취소 등
DELETE FROM statutes_articles WHERE id IN (75, 76, 77, 78, 79);

-- 법령ID: 4, 조문: 014500, 항: 000000, 중복 6개
-- 제목: 법정추인
DELETE FROM statutes_articles WHERE id IN (608, 609, 610, 611, 612);

-- 법령ID: 4, 조문: 084000, 항: 000000, 중복 6개
-- 제목: 재판상 이혼원인
DELETE FROM statutes_articles WHERE id IN (1786, 1787, 1788, 1789, 1790);

-- 법령ID: 4, 조문: 092700, 항: 000①00, 중복 6개
-- 제목: 친권의 상실, 일시 정지 또는 일부 제한과 친권자의 지...
DELETE FROM statutes_articles WHERE id IN (1978, 1979, 1980, 1981, 1982);

-- 법령ID: 4, 조문: 095000, 항: 000①00, 중복 6개
-- 제목: 후견감독인의 동의를 필요로 하는 행위
DELETE FROM statutes_articles WHERE id IN (2056, 2057, 2058, 2059, 2060);

-- 법령ID: 7, 조문: 015300, 항: 000000, 중복 6개
-- 제목: 형식적 기재사항
DELETE FROM statutes_articles WHERE id IN (2700, 2701, 2702, 2703, 2704);

-- 법령ID: 7, 조문: 015400, 항: 000000, 중복 6개
-- 제목: 실질적 기재사항
DELETE FROM statutes_articles WHERE id IN (2706, 2707, 2708, 2709, 2710);

-- 법령ID: 7, 조문: 020800, 항: 000①00, 중복 6개
-- 제목: 판결서의 기재사항 등
DELETE FROM statutes_articles WHERE id IN (2840, 2841, 2842, 2843, 2844);

-- 법령ID: 7, 조문: 042400, 항: 000①00, 중복 6개
-- 제목: 절대적 상고이유
DELETE FROM statutes_articles WHERE id IN (3309, 3310, 3311, 3312, 3313);

-- 법령ID: 8, 조문: 000200, 항: 000000, 중복 6개
-- 제목: 용어의 정의
DELETE FROM statutes_articles WHERE id IN (3473, 3474, 3475, 3476, 3477);

-- 법령ID: 8, 조문: 004300, 항: 000000, 중복 6개
-- 제목: 접견의 중지 등
DELETE FROM statutes_articles WHERE id IN (3585, 3586, 3587, 3588, 3589);

-- 법령ID: 8, 조문: 008700, 항: 000②00, 중복 6개
-- 제목: 강제력의 행사
DELETE FROM statutes_articles WHERE id IN (3749, 3750, 3751, 3752, 3753);

-- 법령ID: 8, 조문: 008800, 항: 000①00, 중복 6개
-- 제목: 무기의 사용
DELETE FROM statutes_articles WHERE id IN (3760, 3761, 3762, 3763, 3764);

-- 법령ID: 8, 조문: 009300, 항: 000000, 중복 6개
-- 제목: 징벌
DELETE FROM statutes_articles WHERE id IN (3788, 3789, 3790, 3791, 3792);

-- 법령ID: 9, 조문: 001100, 항: 000①00, 중복 6개
-- 제목: 자비구매물품의 종류 등
DELETE FROM statutes_articles WHERE id IN (3929, 3930, 3931, 3932, 3933);

-- 법령ID: 9, 조문: 011300, 항: 000000, 중복 6개
-- 제목: 전자장비의 종류
DELETE FROM statutes_articles WHERE id IN (4257, 4258, 4259, 4260, 4261);

-- 법령ID: 9, 조문: 015700, 항: 000②00, 중복 6개
-- 제목: 징벌대상자에 대한 출석통지
DELETE FROM statutes_articles WHERE id IN (4419, 4420, 4421, 4422, 4423);

-- 법령ID: 10, 조문: 000400, 항: 000①00, 중복 6개
-- 제목: 참관
DELETE FROM statutes_articles WHERE id IN (4537, 4538, 4539, 4540, 4541);

-- 법령ID: 10, 조문: 012600, 항: 000000, 중복 6개
-- 제목: 가석방 허가 신청의 절차
DELETE FROM statutes_articles WHERE id IN (4791, 4792, 4793, 4794, 4795);

-- 법령ID: 11, 조문: 009400, 항: 000①00, 중복 6개
-- 제목: 정치 관여
DELETE FROM statutes_articles WHERE id IN (5115, 5116, 5117, 5118, 5119);

-- 법령ID: 12, 조문: 005600, 항: 000000, 중복 6개
-- 제목: 가중ㆍ감경의 순서
DELETE FROM statutes_articles WHERE id IN (5244, 5245, 5246, 5247, 5248);

-- 법령ID: 13, 조문: 004200, 항: 000000, 중복 6개
-- 제목: 접견의 중지 등
DELETE FROM statutes_articles WHERE id IN (5936, 5937, 5938, 5939, 5940);

-- 법령ID: 13, 조문: 010000, 항: 000②00, 중복 6개
-- 제목: 강제력의 행사
DELETE FROM statutes_articles WHERE id IN (6160, 6161, 6162, 6163, 6164);

-- 법령ID: 13, 조문: 010100, 항: 000①00, 중복 6개
-- 제목: 무기의 사용
DELETE FROM statutes_articles WHERE id IN (6171, 6172, 6173, 6174, 6175);

-- 법령ID: 13, 조문: 010700, 항: 000000, 중복 6개
-- 제목: 징벌
DELETE FROM statutes_articles WHERE id IN (6201, 6202, 6203, 6204, 6205);

-- 법령ID: 16, 조문: 003300, 항: 000①00, 중복 6개
-- 제목: 국선변호인
DELETE FROM statutes_articles WHERE id IN (6750, 6751, 6752, 6753, 6754);

-- 법령ID: 16, 조문: 009500, 항: 000000, 중복 6개
-- 제목: 필요적 보석
DELETE FROM statutes_articles WHERE id IN (6931, 6932, 6933, 6934, 6935);

-- 법령ID: 16, 조문: 032700, 항: 000000, 중복 6개
-- 제목: 공소기각의 판결
DELETE FROM statutes_articles WHERE id IN (7745, 7746, 7747, 7748, 7749);

-- 법령ID: 2, 조문: 000400, 항: 000②00, 중복 5개
-- 제목: 난민인정 신청에 필요한 사항의 게시 방법 등
DELETE FROM statutes_articles WHERE id IN (145, 146, 147, 148);

-- 법령ID: 4, 조문: 048200, 항: 000②00, 중복 5개
-- 제목: 변제자대위의 효과, 대위자간의 관계
DELETE FROM statutes_articles WHERE id IN (1171, 1172, 1173, 1174);

-- 법령ID: 4, 조문: 090800, 항: 000①00, 중복 5개
-- 제목: 친양자 입양의 요건 등
DELETE FROM statutes_articles WHERE id IN (1908, 1909, 1910, 1911);

-- 법령ID: 4, 조문: 100400, 항: 000000, 중복 5개
-- 제목: 상속인의 결격사유
DELETE FROM statutes_articles WHERE id IN (2179, 2180, 2181, 2182);

-- 법령ID: 7, 조문: 004100, 항: 000000, 중복 5개
-- 제목: 제척의 이유
DELETE FROM statutes_articles WHERE id IN (2463, 2464, 2465, 2466);

-- 법령ID: 7, 조문: 014000, 항: 000①00, 중복 5개
-- 제목: 법원의 석명처분
DELETE FROM statutes_articles WHERE id IN (2665, 2666, 2667, 2668);

-- 법령ID: 7, 조문: 034400, 항: 000①00, 중복 5개
-- 제목: 문서의 제출의무
DELETE FROM statutes_articles WHERE id IN (3157, 3158, 3159, 3160);

-- 법령ID: 7, 조문: 034500, 항: 000000, 중복 5개
-- 제목: 문서제출신청의 방식
DELETE FROM statutes_articles WHERE id IN (3164, 3165, 3166, 3167);

-- 법령ID: 8, 조문: 001700, 항: 000000, 중복 5개
-- 제목: 고지사항
DELETE FROM statutes_articles WHERE id IN (3512, 3513, 3514, 3515);

-- 법령ID: 8, 조문: 002500, 항: 000①00, 중복 5개
-- 제목: 휴대금품의 영치 등
DELETE FROM statutes_articles WHERE id IN (3531, 3532, 3533, 3534);

-- 법령ID: 9, 조문: 003000, 항: 000②00, 중복 5개
-- 제목: 방송프로그램
DELETE FROM statutes_articles WHERE id IN (3991, 3992, 3993, 3994);

-- 법령ID: 9, 조문: 006900, 항: 000000, 중복 5개
-- 제목: 교화프로그램의 종류
DELETE FROM statutes_articles WHERE id IN (4110, 4111, 4112, 4113);

-- 법령ID: 9, 조문: 007800, 항: 000①00, 중복 5개
-- 제목: 외부통근작업자의 선정기준
DELETE FROM statutes_articles WHERE id IN (4135, 4136, 4137, 4138);

-- 법령ID: 9, 조문: 010100, 항: 000000, 중복 5개
-- 제목: 귀휴조건
DELETE FROM statutes_articles WHERE id IN (4221, 4222, 4223, 4224);

-- 법령ID: 10, 조문: 013900, 항: 000②00, 중복 5개
-- 제목: 권한의 위임
DELETE FROM statutes_articles WHERE id IN (4828, 4829, 4830, 4831);

-- 법령ID: 11, 조문: 003500, 항: 000000, 중복 5개
-- 제목: 근무 태만
DELETE FROM statutes_articles WHERE id IN (4927, 4928, 4929, 4930);

-- 법령ID: 13, 조문: 001700, 항: 000000, 중복 5개
-- 제목: 고지사항
DELETE FROM statutes_articles WHERE id IN (5860, 5861, 5862, 5863);

-- 법령ID: 13, 조문: 002500, 항: 000①00, 중복 5개
-- 제목: 휴대금품의 보관 등
DELETE FROM statutes_articles WHERE id IN (5880, 5881, 5882, 5883);

-- 법령ID: 14, 조문: 005400, 항: 000000, 중복 5개
-- 제목: 간호사의 의료행위
DELETE FROM statutes_articles WHERE id IN (6423, 6424, 6425, 6426);

-- 법령ID: 16, 조문: 005900, 항: 000①00, 중복 5개
-- 제목: 확정 판결서등의 열람ㆍ복사
DELETE FROM statutes_articles WHERE id IN (6845, 6846, 6847, 6848);

-- 법령ID: 16, 조문: 010200, 항: 000②00, 중복 5개
-- 제목: 보석조건의 변경과 취소 등
DELETE FROM statutes_articles WHERE id IN (6969, 6970, 6971, 6972);

-- 법령ID: 1, 조문: 001900, 항: 000000, 중복 4개
-- 제목: 난민인정의 제한
DELETE FROM statutes_articles WHERE id IN (60, 61, 62);

-- 법령ID: 1, 조문: 002600, 항: 000①00, 중복 4개
-- 제목: 위원의 임명
DELETE FROM statutes_articles WHERE id IN (89, 90, 91);

-- 법령ID: 2, 조문: 001200, 항: 000①00, 중복 4개
-- 제목: 위원의 제척과 회피
DELETE FROM statutes_articles WHERE id IN (184, 185, 186);

-- 법령ID: 2, 조문: 001200, 항: 000③00, 중복 4개
-- 제목: 난민위원회의 구성 및 운영 등
DELETE FROM statutes_articles WHERE id IN (177, 178, 179);

-- 법령ID: 3, 조문: 002300, 항: 000②00, 중복 4개
-- 제목: 난민지원시설
DELETE FROM statutes_articles WHERE id IN (305, 306, 307);

-- 법령ID: 4, 조문: 006700, 항: 000000, 중복 4개
-- 제목: 감사의 직무
DELETE FROM statutes_articles WHERE id IN (473, 474, 475);

-- 법령ID: 4, 조문: 016400, 항: 000000, 중복 4개
-- 제목: 1년의 단기소멸시효
DELETE FROM statutes_articles WHERE id IN (651, 652, 653);

-- 법령ID: 4, 조문: 044200, 항: 000①00, 중복 4개
-- 제목: 수탁보증인의 사전구상권
DELETE FROM statutes_articles WHERE id IN (1102, 1103, 1104);

-- 법령ID: 4, 조문: 047700, 항: 000000, 중복 4개
-- 제목: 법정변제충당
DELETE FROM statutes_articles WHERE id IN (1160, 1161, 1162);

-- 법령ID: 4, 조문: 061900, 항: 000000, 중복 4개
-- 제목: 처분능력, 권한없는 자의 할 수 있는 단기임대차
DELETE FROM statutes_articles WHERE id IN (1384, 1385, 1386);

-- 법령ID: 4, 조문: 071700, 항: 000000, 중복 4개
-- 제목: 비임의 탈퇴
DELETE FROM statutes_articles WHERE id IN (1554, 1555, 1556);

-- 법령ID: 4, 조문: 081500, 항: 000000, 중복 4개
-- 제목: 혼인의 무효
DELETE FROM statutes_articles WHERE id IN (1719, 1720, 1721);

-- 법령ID: 4, 조문: 090500, 항: 000000, 중복 4개
-- 제목: 재판상 파양의 원인
DELETE FROM statutes_articles WHERE id IN (1897, 1898, 1899);

-- 법령ID: 4, 조문: 100000, 항: 000①00, 중복 4개
-- 제목: 상속의 순위
DELETE FROM statutes_articles WHERE id IN (2169, 2170, 2171);

-- 법령ID: 4, 조문: 111200, 항: 000000, 중복 4개
-- 제목: 유류분의 권리자와 유류분
DELETE FROM statutes_articles WHERE id IN (2372, 2373, 2374);

-- 법령ID: 7, 조문: 009000, 항: 000②00, 중복 4개
-- 제목: 소송대리권의 범위
DELETE FROM statutes_articles WHERE id IN (2569, 2570, 2571);

-- 법령ID: 7, 조문: 009500, 항: 000000, 중복 4개
-- 제목: 소송대리권이 소멸되지 아니하는 경우
DELETE FROM statutes_articles WHERE id IN (2578, 2579, 2580);

-- 법령ID: 7, 조문: 012900, 항: 000①00, 중복 4개
-- 제목: 구조의 객관적 범위
DELETE FROM statutes_articles WHERE id IN (2639, 2640, 2641);

-- 법령ID: 7, 조문: 021700, 항: 000①00, 중복 4개
-- 제목: 외국재판의 승인
DELETE FROM statutes_articles WHERE id IN (2869, 2870, 2871);

-- 법령ID: 7, 조문: 037700, 항: 000①00, 중복 4개
-- 제목: 신청의 방식
DELETE FROM statutes_articles WHERE id IN (3225, 3226, 3227);

-- 법령ID: 7, 조문: 047900, 항: 000②00, 중복 4개
-- 제목: 공시최고의 기재사항
DELETE FROM statutes_articles WHERE id IN (3422, 3423, 3424);

-- 법령ID: 8, 조문: 001400, 항: 000①00, 중복 4개
-- 제목: 혼거수용
DELETE FROM statutes_articles WHERE id IN (3503, 3504, 3505);

-- 법령ID: 8, 조문: 004200, 항: 000①00, 중복 4개
-- 제목: 접견
DELETE FROM statutes_articles WHERE id IN (3576, 3577, 3578);

-- 법령ID: 8, 조문: 004400, 항: 000④00, 중복 4개
-- 제목: 서신수수
DELETE FROM statutes_articles WHERE id IN (3596, 3597, 3598);

-- 법령ID: 8, 조문: 006600, 항: 000①00, 중복 4개
-- 제목: 귀휴
DELETE FROM statutes_articles WHERE id IN (3667, 3668, 3669);

-- 법령ID: 8, 조문: 008400, 항: 000①00, 중복 4개
-- 제목: 보호장비의 사용
DELETE FROM statutes_articles WHERE id IN (3721, 3722, 3723);

-- 법령ID: 8, 조문: 008500, 항: 000②00, 중복 4개
-- 제목: 보호장비의 종류 및 사용요건
DELETE FROM statutes_articles WHERE id IN (3735, 3736, 3737);

-- 법령ID: 8, 조문: 009200, 항: 000000, 중복 4개
-- 제목: 포상
DELETE FROM statutes_articles WHERE id IN (3784, 3785, 3786);

-- 법령ID: 8, 조문: 010100, 항: 000②00, 중복 4개
-- 제목: 소장 면담
DELETE FROM statutes_articles WHERE id IN (3840, 3841, 3842);

-- 법령ID: 9, 조문: 001900, 항: 000000, 중복 4개
-- 제목: 종교행사의 종류
DELETE FROM statutes_articles WHERE id IN (3962, 3963, 3964);

-- 법령ID: 9, 조문: 002100, 항: 000000, 중복 4개
-- 제목: 종교행사의 참석대상
DELETE FROM statutes_articles WHERE id IN (3968, 3969, 3970);

-- 법령ID: 9, 조문: 002300, 항: 000000, 중복 4개
-- 제목: 종교물품 등의 개인 소지
DELETE FROM statutes_articles WHERE id IN (3974, 3975, 3976);

-- 법령ID: 9, 조문: 003700, 항: 000000, 중복 4개
-- 제목: 처우등급
DELETE FROM statutes_articles WHERE id IN (4036, 4037, 4038);

-- 법령ID: 9, 조문: 004800, 항: 000000, 중복 4개
-- 제목: 접견
DELETE FROM statutes_articles WHERE id IN (4061, 4062, 4063);

-- 법령ID: 9, 조문: 007700, 항: 000②00, 중복 4개
-- 제목: 작업등급
DELETE FROM statutes_articles WHERE id IN (4130, 4131, 4132);

-- 법령ID: 9, 조문: 008400, 항: 000000, 중복 4개
-- 제목: 직업훈련 대상자 선정기준
DELETE FROM statutes_articles WHERE id IN (4148, 4149, 4150);

-- 법령ID: 9, 조문: 008700, 항: 000①00, 중복 4개
-- 제목: 직업훈련의 보류 및 취소 등
DELETE FROM statutes_articles WHERE id IN (4157, 4158, 4159);

-- 법령ID: 9, 조문: 009300, 항: 000②00, 중복 4개
-- 제목: 위원의 임기 등
DELETE FROM statutes_articles WHERE id IN (4201, 4202, 4203);

-- 법령ID: 9, 조문: 011000, 항: 000000, 중복 4개
-- 제목: 교정장비의 종류
DELETE FROM statutes_articles WHERE id IN (4248, 4249, 4250);

-- 법령ID: 9, 조문: 014100, 항: 000000, 중복 4개
-- 제목: 보안장비의 종류별 사용기준
DELETE FROM statutes_articles WHERE id IN (4333, 4334, 4335);

-- 법령ID: 9, 조문: 014200, 항: 000000, 중복 4개
-- 제목: 무기의 종류
DELETE FROM statutes_articles WHERE id IN (4338, 4339, 4340);

-- 법령ID: 9, 조문: 015100, 항: 000000, 중복 4개
-- 제목: 징벌 부과 시 고려사항
DELETE FROM statutes_articles WHERE id IN (4398, 4399, 4400);

-- 법령ID: 9, 조문: 015200, 항: 000000, 중복 4개
-- 제목: 징벌대상자의 조사 시 준수사항
DELETE FROM statutes_articles WHERE id IN (4402, 4403, 4404);

-- 법령ID: 9, 조문: 015300, 항: 000①00, 중복 4개
-- 제목: 조사기간
DELETE FROM statutes_articles WHERE id IN (4406, 4407, 4408);

-- 법령ID: 10, 조문: 006100, 항: 000①00, 중복 4개
-- 제목: 서신 내용의 검열
DELETE FROM statutes_articles WHERE id IN (4655, 4656, 4657);

-- 법령ID: 11, 조문: 006000, 항: 000000, 중복 4개
-- 제목: 군인등에 대한 폭행죄, 협박죄의 특례
DELETE FROM statutes_articles WHERE id IN (5040, 5041, 5042);

-- 법령ID: 12, 조문: 004300, 항: 000①00, 중복 4개
-- 제목: 형의 선고와 자격상실, 자격정지
DELETE FROM statutes_articles WHERE id IN (5207, 5208, 5209);

-- 법령ID: 12, 조문: 005100, 항: 000000, 중복 4개
-- 제목: 양형의 조건
DELETE FROM statutes_articles WHERE id IN (5227, 5228, 5229);

-- 법령ID: 13, 조문: 000200, 항: 000000, 중복 4개
-- 제목: 정의
DELETE FROM statutes_articles WHERE id IN (5805, 5806, 5807);

-- 법령ID: 13, 조문: 001100, 항: 000①00, 중복 4개
-- 제목: 구분수용
DELETE FROM statutes_articles WHERE id IN (5838, 5839, 5840);

-- 법령ID: 13, 조문: 004100, 항: 000①00, 중복 4개
-- 제목: 접견
DELETE FROM statutes_articles WHERE id IN (5923, 5924, 5925);

-- 법령ID: 13, 조문: 004300, 항: 000④00, 중복 4개
-- 제목: 편지수수
DELETE FROM statutes_articles WHERE id IN (5947, 5948, 5949);

-- 법령ID: 13, 조문: 005700, 항: 000②00, 중복 4개
-- 제목: 처우
DELETE FROM statutes_articles WHERE id IN (6010, 6011, 6012);

-- 법령ID: 13, 조문: 007100, 항: 000⑤00, 중복 4개
-- 제목: 작업시간 등
DELETE FROM statutes_articles WHERE id IN (6058, 6059, 6060);

-- 법령ID: 13, 조문: 007700, 항: 000①00, 중복 4개
-- 제목: 귀휴
DELETE FROM statutes_articles WHERE id IN (6074, 6075, 6076);

-- 법령ID: 13, 조문: 009200, 항: 000①00, 중복 4개
-- 제목: 금지물품
DELETE FROM statutes_articles WHERE id IN (6106, 6107, 6108);

-- 법령ID: 13, 조문: 009700, 항: 000①00, 중복 4개
-- 제목: 보호장비의 사용
DELETE FROM statutes_articles WHERE id IN (6132, 6133, 6134);

-- 법령ID: 13, 조문: 009800, 항: 000②00, 중복 4개
-- 제목: 보호장비의 종류 및 사용요건
DELETE FROM statutes_articles WHERE id IN (6146, 6147, 6148);

-- 법령ID: 13, 조문: 010600, 항: 000000, 중복 4개
-- 제목: 포상
DELETE FROM statutes_articles WHERE id IN (6197, 6198, 6199);

-- 법령ID: 13, 조문: 011200, 항: 000④00, 중복 4개
-- 제목: 징벌의 집행
DELETE FROM statutes_articles WHERE id IN (6240, 6241, 6242);

-- 법령ID: 13, 조문: 011600, 항: 000②00, 중복 4개
-- 제목: 소장 면담
DELETE FROM statutes_articles WHERE id IN (6256, 6257, 6258);

-- 법령ID: 14, 조문: 006500, 항: 000①00, 중복 4개
-- 제목: 편지 내용물의 확인
DELETE FROM statutes_articles WHERE id IN (6465, 6466, 6467);

-- 법령ID: 14, 조문: 006600, 항: 000①00, 중복 4개
-- 제목: 편지 내용의 검열
DELETE FROM statutes_articles WHERE id IN (6470, 6471, 6472);

-- 법령ID: 14, 조문: 014500, 항: 000000, 중복 4개
-- 제목: 증명서의 발급
DELETE FROM statutes_articles WHERE id IN (6648, 6649, 6650);

-- 법령ID: 16, 조문: 001100, 항: 000000, 중복 4개
-- 제목: 관련사건의 정의
DELETE FROM statutes_articles WHERE id IN (6687, 6688, 6689);

-- 법령ID: 16, 조문: 009900, 항: 000①00, 중복 4개
-- 제목: 보석조건의 결정 시 고려사항
DELETE FROM statutes_articles WHERE id IN (6951, 6952, 6953);

-- 법령ID: 16, 조문: 019400, 항: 000②00, 중복 4개
-- 제목: 무죄판결과 비용보상
DELETE FROM statutes_articles WHERE id IN (7183, 7184, 7185);

-- 법령ID: 16, 조문: 020000, 항: 000④00, 중복 4개
-- 제목: 긴급체포와 영장청구기간
DELETE FROM statutes_articles WHERE id IN (7240, 7241, 7242);

-- 법령ID: 16, 조문: 021100, 항: 000②00, 중복 4개
-- 제목: 현행범인과 준현행범인
DELETE FROM statutes_articles WHERE id IN (7276, 7277, 7278);

-- 법령ID: 16, 조문: 021400, 항: 000②00, 중복 4개
-- 제목: 재체포 및 재구속의 제한
DELETE FROM statutes_articles WHERE id IN (7304, 7305, 7306);

-- 법령ID: 16, 조문: 024400, 항: 000①00, 중복 4개
-- 제목: 진술거부권 등의 고지
DELETE FROM statutes_articles WHERE id IN (7389, 7390, 7391);

-- 법령ID: 16, 조문: 025400, 항: 000③00, 중복 4개
-- 제목: 공소제기의 방식과 공소장
DELETE FROM statutes_articles WHERE id IN (7450, 7451, 7452);

-- 법령ID: 16, 조문: 026600, 항: 000①00, 중복 4개
-- 제목: 공소제기 후 검사가 보관하고 있는 서류 등의 열람ㆍ등사
DELETE FROM statutes_articles WHERE id IN (7497, 7498, 7499);

-- 법령ID: 16, 조문: 026600, 항: 000①00, 중복 4개
-- 제목: 피고인 또는 변호인이 보관하고 있는 서류등의 열람ㆍ등사
DELETE FROM statutes_articles WHERE id IN (7543, 7544, 7545);

-- 법령ID: 16, 조문: 027700, 항: 000000, 중복 4개
-- 제목: 경미사건 등과 피고인의 불출석
DELETE FROM statutes_articles WHERE id IN (7593, 7594, 7595);

-- 법령ID: 16, 조문: 032600, 항: 000000, 중복 4개
-- 제목: 면소의 판결
DELETE FROM statutes_articles WHERE id IN (7741, 7742, 7743);

-- 법령ID: 16, 조문: 032800, 항: 000①00, 중복 4개
-- 제목: 공소기각의 결정
DELETE FROM statutes_articles WHERE id IN (7751, 7752, 7753);

-- 법령ID: 16, 조문: 038300, 항: 000000, 중복 4개
-- 제목: 상고이유
DELETE FROM statutes_articles WHERE id IN (7876, 7877, 7878);

-- 법령ID: 16, 조문: 041600, 항: 000①00, 중복 4개
-- 제목: 준항고
DELETE FROM statutes_articles WHERE id IN (7926, 7927, 7928);

-- 법령ID: 16, 조문: 042400, 항: 000000, 중복 4개
-- 제목: 재심청구권자
DELETE FROM statutes_articles WHERE id IN (7949, 7950, 7951);

-- 법령ID: 1, 조문: 000800, 항: 000⑤00, 중복 3개
-- 제목: 난민인정 심사
DELETE FROM statutes_articles WHERE id IN (34, 35);

-- 법령ID: 1, 조문: 002900, 항: 000②00, 중복 3개
-- 제목: 유엔난민기구와의 교류ㆍ협력
DELETE FROM statutes_articles WHERE id IN (101, 102);

-- 법령ID: 1, 조문: 002900, 항: 000①00, 중복 3개
-- 제목: 유엔난민기구와의 교류ㆍ협력
DELETE FROM statutes_articles WHERE id IN (98, 99);

-- 법령ID: 2, 조문: 000200, 항: 000①00, 중복 3개
-- 제목: 난민인정 신청의 방법과 절차 등
DELETE FROM statutes_articles WHERE id IN (138, 139);

-- 법령ID: 2, 조문: 000800, 항: 000③00, 중복 3개
-- 제목: 난민인정증명서 등
DELETE FROM statutes_articles WHERE id IN (162, 163);

-- 법령ID: 2, 조문: 001000, 항: 000①00, 중복 3개
-- 제목: 이의신청 절차 등
DELETE FROM statutes_articles WHERE id IN (168, 169);

-- 법령ID: 3, 조문: 001100, 항: 000①00, 중복 3개
-- 제목: 이의신청에 대한 결정 등
DELETE FROM statutes_articles WHERE id IN (271, 272);

-- 법령ID: 3, 조문: 002100, 항: 000000, 중복 3개
-- 제목: 특정 난민신청자의 처우 제한
DELETE FROM statutes_articles WHERE id IN (300, 301);

-- 법령ID: 4, 조문: 008500, 항: 000①00, 중복 3개
-- 제목: 해산등기
DELETE FROM statutes_articles WHERE id IN (505, 506);

-- 법령ID: 4, 조문: 008700, 항: 000①00, 중복 3개
-- 제목: 청산인의 직무
DELETE FROM statutes_articles WHERE id IN (511, 512);

-- 법령ID: 4, 조문: 016800, 항: 000000, 중복 3개
-- 제목: 소멸시효의 중단사유
DELETE FROM statutes_articles WHERE id IN (661, 662);

-- 법령ID: 4, 조문: 028000, 항: 000①00, 중복 3개
-- 제목: 존속기간을 약정한 지상권
DELETE FROM statutes_articles WHERE id IN (842, 843);

-- 법령ID: 4, 조문: 043600, 항: 000②00, 중복 3개
-- 제목: 채권자의 정보제공의무와 통지의무 등
DELETE FROM statutes_articles WHERE id IN (1091, 1092);

-- 법령ID: 4, 조문: 051100, 항: 000000, 중복 3개
-- 제목: 약식배서의 처리방식
DELETE FROM statutes_articles WHERE id IN (1219, 1220);

-- 법령ID: 4, 조문: 077700, 항: 000000, 중복 3개
-- 제목: 친족의 범위
DELETE FROM statutes_articles WHERE id IN (1654, 1655);

-- 법령ID: 4, 조문: 081600, 항: 000000, 중복 3개
-- 제목: 혼인취소의 사유
DELETE FROM statutes_articles WHERE id IN (1723, 1724);

-- 법령ID: 4, 조문: 083700, 항: 000②00, 중복 3개
-- 제목: 이혼과 자의 양육책임
DELETE FROM statutes_articles WHERE id IN (1768, 1769);

-- 법령ID: 4, 조문: 087000, 항: 000①00, 중복 3개
-- 제목: 미성년자 입양에 대한 부모의 동의
DELETE FROM statutes_articles WHERE id IN (1843, 1844);

-- 법령ID: 4, 조문: 088400, 항: 000①00, 중복 3개
-- 제목: 입양 취소의 원인
DELETE FROM statutes_articles WHERE id IN (1870, 1871);

-- 법령ID: 4, 조문: 090800, 항: 000②00, 중복 3개
-- 제목: 친양자 입양의 요건 등
DELETE FROM statutes_articles WHERE id IN (1913, 1914);

-- 법령ID: 4, 조문: 090900, 항: 000⑤00, 중복 3개
-- 제목: 친권자의 지정 등
DELETE FROM statutes_articles WHERE id IN (1939, 1940);

-- 법령ID: 4, 조문: 092700, 항: 000②00, 중복 3개
-- 제목: 친권의 상실, 일시 정지 또는 일부 제한과 친권자의 지...
DELETE FROM statutes_articles WHERE id IN (1984, 1985);

-- 법령ID: 4, 조문: 094500, 항: 000000, 중복 3개
-- 제목: 미성년자의 신분에 관한 후견인의 권리ㆍ의무
DELETE FROM statutes_articles WHERE id IN (2038, 2039);

-- 법령ID: 4, 조문: 097400, 항: 000000, 중복 3개
-- 제목: 부양의무
DELETE FROM statutes_articles WHERE id IN (2133, 2134);

-- 법령ID: 4, 조문: 102600, 항: 000000, 중복 3개
-- 제목: 법정단순승인
DELETE FROM statutes_articles WHERE id IN (2226, 2227);

-- 법령ID: 4, 조문: 107200, 항: 000①00, 중복 3개
-- 제목: 증인의 결격사유
DELETE FROM statutes_articles WHERE id IN (2307, 2308);

-- 법령ID: 7, 조문: 006200, 항: 000①00, 중복 3개
-- 제목: 제한능력자를 위한 특별대리인
DELETE FROM statutes_articles WHERE id IN (2502, 2503);

-- 법령ID: 7, 조문: 007700, 항: 000000, 중복 3개
-- 제목: 참가인에 대한 재판의 효력
DELETE FROM statutes_articles WHERE id IN (2542, 2543);

-- 법령ID: 7, 조문: 020800, 항: 000③00, 중복 3개
-- 제목: 판결서의 기재사항 등
DELETE FROM statutes_articles WHERE id IN (2847, 2848);

-- 법령ID: 7, 조문: 023100, 항: 000000, 중복 3개
-- 제목: 화해권고결정의 효력
DELETE FROM statutes_articles WHERE id IN (2906, 2907);

-- 법령ID: 7, 조문: 028400, 항: 000①00, 중복 3개
-- 제목: 변론준비절차의 종결
DELETE FROM statutes_articles WHERE id IN (3027, 3028);

-- 법령ID: 7, 조문: 028500, 항: 000①00, 중복 3개
-- 제목: 변론준비기일을 종결한 효과
DELETE FROM statutes_articles WHERE id IN (3031, 3032);

-- 법령ID: 7, 조문: 030900, 항: 000000, 중복 3개
-- 제목: 출석요구서의 기재사항
DELETE FROM statutes_articles WHERE id IN (3072, 3073);

-- 법령ID: 7, 조문: 031300, 항: 000000, 중복 3개
-- 제목: 수명법관ㆍ수탁판사에 의한 증인신문
DELETE FROM statutes_articles WHERE id IN (3088, 3089);

-- 법령ID: 7, 조문: 045800, 항: 000000, 중복 3개
-- 제목: 재심소장의 필수적 기재사항
DELETE FROM statutes_articles WHERE id IN (3380, 3381);

-- 법령ID: 8, 조문: 001300, 항: 000①00, 중복 3개
-- 제목: 독거수용
DELETE FROM statutes_articles WHERE id IN (3499, 3500);

-- 법령ID: 8, 조문: 004200, 항: 000②00, 중복 3개
-- 제목: 접견
DELETE FROM statutes_articles WHERE id IN (3580, 3581);

-- 법령ID: 8, 조문: 004400, 항: 000①00, 중복 3개
-- 제목: 서신수수
DELETE FROM statutes_articles WHERE id IN (3591, 3592);

-- 법령ID: 8, 조문: 007900, 항: 000000, 중복 3개
-- 제목: 금지물품
DELETE FROM statutes_articles WHERE id IN (3697, 3698);

-- 법령ID: 9, 조문: 001200, 항: 000①00, 중복 3개
-- 제목: 구매허가 및 신청제한
DELETE FROM statutes_articles WHERE id IN (3937, 3938);

-- 법령ID: 9, 조문: 003000, 항: 000③00, 중복 3개
-- 제목: 방송프로그램
DELETE FROM statutes_articles WHERE id IN (3996, 3997);

-- 법령ID: 9, 조문: 003300, 항: 000000, 중복 3개
-- 제목: 심사의 종류
DELETE FROM statutes_articles WHERE id IN (4021, 4022);

-- 법령ID: 9, 조문: 003400, 항: 000①00, 중복 3개
-- 제목: 정기심사
DELETE FROM statutes_articles WHERE id IN (4024, 4025);

-- 법령ID: 9, 조문: 003600, 항: 000①00, 중복 3개
-- 제목: 특별심사
DELETE FROM statutes_articles WHERE id IN (4032, 4033);

-- 법령ID: 9, 조문: 005600, 항: 000①00, 중복 3개
-- 제목: 교육대상자 선발 취소 등
DELETE FROM statutes_articles WHERE id IN (4082, 4083);

-- 법령ID: 9, 조문: 005700, 항: 000000, 중복 3개
-- 제목: 교육ㆍ훈련과정
DELETE FROM statutes_articles WHERE id IN (4086, 4087);

-- 법령ID: 9, 조문: 005900, 항: 000②00, 중복 3개
-- 제목: 방송통신고등학교 교육과정 설치 및 운영
DELETE FROM statutes_articles WHERE id IN (4093, 4094);

-- 법령ID: 9, 조문: 008500, 항: 000000, 중복 3개
-- 제목: 직업훈련 대상자 선정의 제한
DELETE FROM statutes_articles WHERE id IN (4152, 4153);

-- 법령ID: 9, 조문: 012500, 항: 000①00, 중복 3개
-- 제목: 수갑의 사용방법
DELETE FROM statutes_articles WHERE id IN (4290, 4291);

-- 법령ID: 9, 조문: 013200, 항: 000①00, 중복 3개
-- 제목: 포승의 사용방법
DELETE FROM statutes_articles WHERE id IN (4307, 4308);

-- 법령ID: 9, 조문: 016600, 항: 000①00, 중복 3개
-- 제목: 위원회의 구성 등
DELETE FROM statutes_articles WHERE id IN (4444, 4445);

-- 법령ID: 9, 조문: 018200, 항: 000000, 중복 3개
-- 제목: 사회의 감정에 대한 심사
DELETE FROM statutes_articles WHERE id IN (4501, 4502);

-- 법령ID: 9, 조문: 018300, 항: 000000, 중복 3개
-- 제목: 군에 대한 피해정도에 관한 심사
DELETE FROM statutes_articles WHERE id IN (4504, 4505);

-- 법령ID: 10, 조문: 004700, 항: 000000, 중복 3개
-- 제목: 실외운동
DELETE FROM statutes_articles WHERE id IN (4614, 4615);

-- 법령ID: 10, 조문: 006000, 항: 000①00, 중복 3개
-- 제목: 서신 내용물의 확인
DELETE FROM statutes_articles WHERE id IN (4651, 4652);

-- 법령ID: 10, 조문: 011400, 항: 000①00, 중복 3개
-- 제목: 징벌위원회 외부 위원
DELETE FROM statutes_articles WHERE id IN (4761, 4762);

-- 법령ID: 11, 조문: 000100, 항: 000③00, 중복 3개
-- 제목: 적용대상자
DELETE FROM statutes_articles WHERE id IN (4836, 4837);

-- 법령ID: 11, 조문: 000500, 항: 000000, 중복 3개
-- 제목: 반란
DELETE FROM statutes_articles WHERE id IN (4864, 4865);

-- 법령ID: 11, 조문: 001300, 항: 000③00, 중복 3개
-- 제목: 간첩
DELETE FROM statutes_articles WHERE id IN (4880, 4881);

-- 법령ID: 11, 조문: 002400, 항: 000000, 중복 3개
-- 제목: 직무유기
DELETE FROM statutes_articles WHERE id IN (4903, 4904);

-- 법령ID: 11, 조문: 002700, 항: 000000, 중복 3개
-- 제목: 지휘관의 수소 이탈
DELETE FROM statutes_articles WHERE id IN (4909, 4910);

-- 법령ID: 11, 조문: 002800, 항: 000000, 중복 3개
-- 제목: 초병의 수소 이탈
DELETE FROM statutes_articles WHERE id IN (4912, 4913);

-- 법령ID: 11, 조문: 003000, 항: 000①00, 중복 3개
-- 제목: 군무 이탈
DELETE FROM statutes_articles WHERE id IN (4917, 4918);

-- 법령ID: 11, 조문: 003600, 항: 000000, 중복 3개
-- 제목: 비행군기 문란
DELETE FROM statutes_articles WHERE id IN (4932, 4933);

-- 법령ID: 11, 조문: 003800, 항: 000①00, 중복 3개
-- 제목: 거짓 명령, 통보, 보고
DELETE FROM statutes_articles WHERE id IN (4937, 4938);

-- 법령ID: 11, 조문: 004000, 항: 000①00, 중복 3개
-- 제목: 초령 위반
DELETE FROM statutes_articles WHERE id IN (4942, 4943);

-- 법령ID: 11, 조문: 004400, 항: 000000, 중복 3개
-- 제목: 항명
DELETE FROM statutes_articles WHERE id IN (4956, 4957);

-- 법령ID: 11, 조문: 004500, 항: 000000, 중복 3개
-- 제목: 집단 항명
DELETE FROM statutes_articles WHERE id IN (4959, 4960);

-- 법령ID: 11, 조문: 005200, 항: 000①00, 중복 3개
-- 제목: 상관에 대한 폭행치사상
DELETE FROM statutes_articles WHERE id IN (4973, 4974);

-- 법령ID: 11, 조문: 005200, 항: 000000, 중복 3개
-- 제목: 상관에 대한 중상해
DELETE FROM statutes_articles WHERE id IN (4985, 4986);

-- 법령ID: 11, 조문: 005200, 항: 000000, 중복 3개
-- 제목: 상관에 대한 상해치사
DELETE FROM statutes_articles WHERE id IN (4988, 4989);

-- 법령ID: 11, 조문: 005800, 항: 000①00, 중복 3개
-- 제목: 초병에 대한 폭행치사상
DELETE FROM statutes_articles WHERE id IN (5001, 5002);

-- 법령ID: 11, 조문: 005800, 항: 000000, 중복 3개
-- 제목: 초병에 대한 상해치사
DELETE FROM statutes_articles WHERE id IN (5015, 5016);

-- 법령ID: 11, 조문: 006000, 항: 000000, 중복 3개
-- 제목: 직무수행 중인 군인등에 대한 상해치사
DELETE FROM statutes_articles WHERE id IN (5037, 5038);

-- 법령ID: 11, 조문: 006000, 항: 000④00, 중복 3개
-- 제목: 직무수행 중인 군인등에 대한 폭행, 협박 등
DELETE FROM statutes_articles WHERE id IN (5025, 5026);

-- 법령ID: 11, 조문: 006100, 항: 000000, 중복 3개
-- 제목: 특수소요
DELETE FROM statutes_articles WHERE id IN (5044, 5045);

-- 법령ID: 11, 조문: 007800, 항: 000000, 중복 3개
-- 제목: 초소 침범
DELETE FROM statutes_articles WHERE id IN (5079, 5080);

-- 법령ID: 11, 조문: 008100, 항: 000000, 중복 3개
-- 제목: 암호 부정사용
DELETE FROM statutes_articles WHERE id IN (5085, 5086);

-- 법령ID: 12, 조문: 003800, 항: 000①00, 중복 3개
-- 제목: 경합범과 처벌례
DELETE FROM statutes_articles WHERE id IN (5187, 5188);

-- 법령ID: 12, 조문: 004800, 항: 000①00, 중복 3개
-- 제목: 몰수의 대상과 추징
DELETE FROM statutes_articles WHERE id IN (5217, 5218);

-- 법령ID: 12, 조문: 008700, 항: 000000, 중복 3개
-- 제목: 내란
DELETE FROM statutes_articles WHERE id IN (5315, 5316);

-- 법령ID: 13, 조문: 001200, 항: 000①00, 중복 3개
-- 제목: 구분수용의 예외
DELETE FROM statutes_articles WHERE id IN (5843, 5844);

-- 법령ID: 13, 조문: 001400, 항: 000000, 중복 3개
-- 제목: 독거수용
DELETE FROM statutes_articles WHERE id IN (5851, 5852);

-- 법령ID: 13, 조문: 004100, 항: 000④00, 중복 3개
-- 제목: 접견
DELETE FROM statutes_articles WHERE id IN (5931, 5932);

-- 법령ID: 13, 조문: 004300, 항: 000①00, 중복 3개
-- 제목: 편지수수
DELETE FROM statutes_articles WHERE id IN (5942, 5943);

-- 법령ID: 13, 조문: 005300, 항: 000①00, 중복 3개
-- 제목: 유아의 양육
DELETE FROM statutes_articles WHERE id IN (5993, 5994);

-- 법령ID: 13, 조문: 012000, 항: 000③00, 중복 3개
-- 제목: 위원회의 구성
DELETE FROM statutes_articles WHERE id IN (6277, 6278);

-- 법령ID: 14, 조문: 000100, 항: 000②00, 중복 3개
-- 제목: 협의체의 구성 및 운영 등
DELETE FROM statutes_articles WHERE id IN (6325, 6326);

-- 법령ID: 14, 조문: 002200, 항: 000①00, 중복 3개
-- 제목: 지방교정청장의 이송승인권
DELETE FROM statutes_articles WHERE id IN (6366, 6367);

-- 법령ID: 14, 조문: 004900, 항: 000000, 중복 3개
-- 제목: 실외운동
DELETE FROM statutes_articles WHERE id IN (6411, 6412);

-- 법령ID: 14, 조문: 005900, 항: 000②00, 중복 3개
-- 제목: 접견의 예외
DELETE FROM statutes_articles WHERE id IN (6439, 6440);

-- 법령ID: 14, 조문: 005900, 항: 000③00, 중복 3개
-- 제목: 접견의 예외
DELETE FROM statutes_articles WHERE id IN (6442, 6443);

-- 법령ID: 14, 조문: 008100, 항: 000④00, 중복 3개
-- 제목: 노인수용자 등의 정의
DELETE FROM statutes_articles WHERE id IN (6503, 6504);

-- 법령ID: 14, 조문: 008300, 항: 000000, 중복 3개
-- 제목: 경비등급별 설비 및 계호
DELETE FROM statutes_articles WHERE id IN (6509, 6510);

-- 법령ID: 14, 조문: 012800, 항: 000000, 중복 3개
-- 제목: 포상금의 환수
DELETE FROM statutes_articles WHERE id IN (6584, 6585);

-- 법령ID: 16, 조문: 007000, 항: 000①00, 중복 3개
-- 제목: 구속의 사유
DELETE FROM statutes_articles WHERE id IN (6878, 6879);

-- 법령ID: 16, 조문: 016500, 항: 000①00, 중복 3개
-- 제목: 비디오 등 중계장치 등에 의한 증인신문
DELETE FROM statutes_articles WHERE id IN (7109, 7110);

-- 법령ID: 16, 조문: 026000, 항: 000②00, 중복 3개
-- 제목: 재정신청
DELETE FROM statutes_articles WHERE id IN (7466, 7467);

-- 법령ID: 16, 조문: 026600, 항: 000000, 중복 3개
-- 제목: 공판준비절차의 종결사유
DELETE FROM statutes_articles WHERE id IN (7551, 7552);

-- 법령ID: 16, 조문: 029400, 항: 000①00, 중복 3개
-- 제목: 피해자등의 진술권
DELETE FROM statutes_articles WHERE id IN (7650, 7651);

-- 법령ID: 16, 조문: 031500, 항: 000000, 중복 3개
-- 제목: 당연히 증거능력이 있는 서류
DELETE FROM statutes_articles WHERE id IN (7714, 7715);

-- 법령ID: 1, 조문: 000500, 항: 000②00, 중복 2개
-- 제목: 난민인정 신청
DELETE FROM statutes_articles WHERE id IN (16);

-- 법령ID: 1, 조문: 004700, 항: 000000, 중복 2개
-- 제목: 벌칙
DELETE FROM statutes_articles WHERE id IN (135);

-- 법령ID: 2, 조문: 000700, 항: 000④00, 중복 2개
-- 제목: 열람ㆍ복사 신청 등
DELETE FROM statutes_articles WHERE id IN (156);

-- 법령ID: 2, 조문: 001300, 항: 000①00, 중복 2개
-- 제목: 교육비 지원 추천 절차
DELETE FROM statutes_articles WHERE id IN (197);

-- 법령ID: 2, 조문: 001600, 항: 000①00, 중복 2개
-- 제목: 의료 지원 절차
DELETE FROM statutes_articles WHERE id IN (210);

-- 법령ID: 3, 조문: 000200, 항: 000①00, 중복 2개
-- 제목: 인도적 체류 허가
DELETE FROM statutes_articles WHERE id IN (217);

-- 법령ID: 3, 조문: 000600, 항: 000000, 중복 2개
-- 제목: 법무부 내 난민전담공무원의 요건
DELETE FROM statutes_articles WHERE id IN (242);

-- 법령ID: 3, 조문: 000600, 항: 000000, 중복 2개
-- 제목: 난민심사관의 자격
DELETE FROM statutes_articles WHERE id IN (240);

-- 법령ID: 3, 조문: 000700, 항: 000⑥00, 중복 2개
-- 제목: 난민심사관 등의 업무 수행
DELETE FROM statutes_articles WHERE id IN (249);

-- 법령ID: 3, 조문: 000800, 항: 000③00, 중복 2개
-- 제목: 통역
DELETE FROM statutes_articles WHERE id IN (254);

-- 법령ID: 3, 조문: 000900, 항: 000②00, 중복 2개
-- 제목: 전자민원창구를 통한 이의신청 등의 처리
DELETE FROM statutes_articles WHERE id IN (264);

-- 법령ID: 3, 조문: 001200, 항: 000①00, 중복 2개
-- 제목: 재정착희망난민 국내 정착 허가
DELETE FROM statutes_articles WHERE id IN (276);

-- 법령ID: 4, 조문: 011800, 항: 000000, 중복 2개
-- 제목: 대리권의 범위
DELETE FROM statutes_articles WHERE id IN (573);

-- 법령ID: 4, 조문: 012700, 항: 000000, 중복 2개
-- 제목: 대리권의 소멸사유
DELETE FROM statutes_articles WHERE id IN (585);

-- 법령ID: 4, 조문: 038800, 항: 000000, 중복 2개
-- 제목: 기한의 이익의 상실
DELETE FROM statutes_articles WHERE id IN (1009);

-- 법령ID: 4, 조문: 055600, 항: 000①00, 중복 2개
-- 제목: 수증자의 행위와 증여의 해제
DELETE FROM statutes_articles WHERE id IN (1284);

-- 법령ID: 4, 조문: 063500, 항: 000②00, 중복 2개
-- 제목: 기간의 약정없는 임대차의 해지통고
DELETE FROM statutes_articles WHERE id IN (1410);

-- 법령ID: 4, 조문: 077900, 항: 000①00, 중복 2개
-- 제목: 가족의 범위
DELETE FROM statutes_articles WHERE id IN (1659);

-- 법령ID: 4, 조문: 083600, 항: 000②00, 중복 2개
-- 제목: 이혼의 절차
DELETE FROM statutes_articles WHERE id IN (1762);

-- 법령ID: 4, 조문: 086900, 항: 000③00, 중복 2개
-- 제목: 입양의 의사표시
DELETE FROM statutes_articles WHERE id IN (1839);

-- 법령ID: 4, 조문: 087000, 항: 000②00, 중복 2개
-- 제목: 미성년자 입양에 대한 부모의 동의
DELETE FROM statutes_articles WHERE id IN (1846);

-- 법령ID: 4, 조문: 088300, 항: 000000, 중복 2개
-- 제목: 입양 무효의 원인
DELETE FROM statutes_articles WHERE id IN (1868);

-- 법령ID: 4, 조문: 090800, 항: 000①00, 중복 2개
-- 제목: 친양자의 파양
DELETE FROM statutes_articles WHERE id IN (1921);

-- 법령ID: 7, 조문: 002800, 항: 000①00, 중복 2개
-- 제목: 관할의 지정
DELETE FROM statutes_articles WHERE id IN (2439);

-- 법령ID: 7, 조문: 005500, 항: 000①00, 중복 2개
-- 제목: 제한능력자의 소송능력
DELETE FROM statutes_articles WHERE id IN (2491);

-- 법령ID: 7, 조문: 016300, 항: 000①00, 중복 2개
-- 제목: 비밀보호를 위한 열람 등의 제한
DELETE FROM statutes_articles WHERE id IN (2731);

-- 법령ID: 7, 조문: 022700, 항: 000②00, 중복 2개
-- 제목: 이의신청의 방식
DELETE FROM statutes_articles WHERE id IN (2895);

-- 법령ID: 7, 조문: 031400, 항: 000000, 중복 2개
-- 제목: 증언거부권
DELETE FROM statutes_articles WHERE id IN (3091);

-- 법령ID: 7, 조문: 031500, 항: 000①00, 중복 2개
-- 제목: 증언거부권
DELETE FROM statutes_articles WHERE id IN (3093);

-- 법령ID: 7, 조문: 032200, 항: 000000, 중복 2개
-- 제목: 선서무능력
DELETE FROM statutes_articles WHERE id IN (3106);

-- 법령ID: 7, 조문: 032700, 항: 000①00, 중복 2개
-- 제목: 비디오 등 중계장치에 의한 증인신문
DELETE FROM statutes_articles WHERE id IN (3118);

-- 법령ID: 7, 조문: 033900, 항: 000①00, 중복 2개
-- 제목: 비디오 등 중계장치 등에 의한 감정인신문
DELETE FROM statutes_articles WHERE id IN (3146);

-- 법령ID: 7, 조문: 034400, 항: 000②00, 중복 2개
-- 제목: 문서의 제출의무
DELETE FROM statutes_articles WHERE id IN (3162);

-- 법령ID: 7, 조문: 039700, 항: 000②00, 중복 2개
-- 제목: 항소의 방식, 항소장의 기재사항
DELETE FROM statutes_articles WHERE id IN (3265);

-- 법령ID: 7, 조문: 043700, 항: 000000, 중복 2개
-- 제목: 파기자판
DELETE FROM statutes_articles WHERE id IN (3333);

-- 법령ID: 8, 조문: 001000, 항: 000000, 중복 2개
-- 제목: 구분수용
DELETE FROM statutes_articles WHERE id IN (3492);

-- 법령ID: 8, 조문: 001100, 항: 000①00, 중복 2개
-- 제목: 구분수용의 예외
DELETE FROM statutes_articles WHERE id IN (3494);

-- 법령ID: 8, 조문: 002700, 항: 000①00, 중복 2개
-- 제목: 군수용자에 대한 금품 교부
DELETE FROM statutes_articles WHERE id IN (3541);

-- 법령ID: 8, 조문: 004600, 항: 000③00, 중복 2개
-- 제목: 종교행사의 참석 등
DELETE FROM statutes_articles WHERE id IN (3618);

-- 법령ID: 8, 조문: 004900, 항: 000②00, 중복 2개
-- 제목: 라디오 청취와 텔레비전 시청
DELETE FROM statutes_articles WHERE id IN (3626);

-- 법령ID: 8, 조문: 006400, 항: 000①00, 중복 2개
-- 제목: 위로금ㆍ조위금
DELETE FROM statutes_articles WHERE id IN (3662);

-- 법령ID: 8, 조문: 006600, 항: 000②00, 중복 2개
-- 제목: 귀휴
DELETE FROM statutes_articles WHERE id IN (3671);

-- 법령ID: 8, 조문: 006700, 항: 000000, 중복 2개
-- 제목: 귀휴의 취소
DELETE FROM statutes_articles WHERE id IN (3675);

-- 법령ID: 8, 조문: 008200, 항: 000①00, 중복 2개
-- 제목: 보호실 수용
DELETE FROM statutes_articles WHERE id IN (3709);

-- 법령ID: 8, 조문: 008300, 항: 000①00, 중복 2개
-- 제목: 진정실 수용
DELETE FROM statutes_articles WHERE id IN (3716);

-- 법령ID: 8, 조문: 009500, 항: 000②00, 중복 2개
-- 제목: 징벌의 부과
DELETE FROM statutes_articles WHERE id IN (3809);

-- 법령ID: 8, 조문: 009600, 항: 000②00, 중복 2개
-- 제목: 징벌대상자의 조사
DELETE FROM statutes_articles WHERE id IN (3816);

-- 법령ID: 8, 조문: 011600, 항: 000①00, 중복 2개
-- 제목: 주류의 반입 등
DELETE FROM statutes_articles WHERE id IN (3885);

-- 법령ID: 8, 조문: 011700, 항: 000000, 중복 2개
-- 제목: 출석의무의 위반 등
DELETE FROM statutes_articles WHERE id IN (3889);

-- 법령ID: 9, 조문: 002500, 항: 000000, 중복 2개
-- 제목: 구독허가의 취소
DELETE FROM statutes_articles WHERE id IN (3980);

-- 법령ID: 9, 조문: 003500, 항: 000000, 중복 2개
-- 제목: 형기의 3분의 1 심사
DELETE FROM statutes_articles WHERE id IN (4030);

-- 법령ID: 9, 조문: 005100, 항: 000①00, 중복 2개
-- 제목: 전화통화 등
DELETE FROM statutes_articles WHERE id IN (4070);

-- 법령ID: 9, 조문: 005800, 항: 000①00, 중복 2개
-- 제목: 검정고시반 설치 및 운영
DELETE FROM statutes_articles WHERE id IN (4089);

-- 법령ID: 9, 조문: 008900, 항: 000②00, 중복 2개
-- 제목: 귀휴 심사
DELETE FROM statutes_articles WHERE id IN (4175);

-- 법령ID: 9, 조문: 009700, 항: 000①00, 중복 2개
-- 제목: 귀휴심사에 필요한 서류
DELETE FROM statutes_articles WHERE id IN (4213);

-- 법령ID: 9, 조문: 012700, 항: 000000, 중복 2개
-- 제목: 발목보호장비의 사용방법
DELETE FROM statutes_articles WHERE id IN (4297);

-- 법령ID: 9, 조문: 012800, 항: 000000, 중복 2개
-- 제목: 보호대의 사용방법
DELETE FROM statutes_articles WHERE id IN (4299);

-- 법령ID: 9, 조문: 014000, 항: 000①00, 중복 2개
-- 제목: 보안장비의 종류별 사용요건
DELETE FROM statutes_articles WHERE id IN (4328);

-- 법령ID: 9, 조문: 014000, 항: 000②00, 중복 2개
-- 제목: 보안장비의 종류별 사용요건
DELETE FROM statutes_articles WHERE id IN (4330);

-- 법령ID: 9, 조문: 014300, 항: 000①00, 중복 2개
-- 제목: 무기의 종류별 사용요건
DELETE FROM statutes_articles WHERE id IN (4342);

-- 법령ID: 9, 조문: 014300, 항: 000②00, 중복 2개
-- 제목: 무기의 종류별 사용요건
DELETE FROM statutes_articles WHERE id IN (4344);

-- 법령ID: 10, 조문: 000600, 항: 000000, 중복 2개
-- 제목: 독거수용의 구분
DELETE FROM statutes_articles WHERE id IN (4548);

-- 법령ID: 10, 조문: 005300, 항: 000④00, 중복 2개
-- 제목: 접견
DELETE FROM statutes_articles WHERE id IN (4633);

-- 법령ID: 10, 조문: 005400, 항: 000②00, 중복 2개
-- 제목: 접견의 예외
DELETE FROM statutes_articles WHERE id IN (4637);

-- 법령ID: 10, 조문: 005700, 항: 000③00, 중복 2개
-- 제목: 접견기록물의 관리 등
DELETE FROM statutes_articles WHERE id IN (4646);

-- 법령ID: 11, 조문: 003200, 항: 000000, 중복 2개
-- 제목: 이탈자 비호
DELETE FROM statutes_articles WHERE id IN (4922);

-- 법령ID: 11, 조문: 003700, 항: 000000, 중복 2개
-- 제목: 위계로 인한 항행 위험
DELETE FROM statutes_articles WHERE id IN (4935);

-- 법령ID: 11, 조문: 004100, 항: 000②00, 중복 2개
-- 제목: 근무 기피 목적의 사술
DELETE FROM statutes_articles WHERE id IN (4948);

-- 법령ID: 11, 조문: 004100, 항: 000①00, 중복 2개
-- 제목: 근무 기피 목적의 사술
DELETE FROM statutes_articles WHERE id IN (4946);

-- 법령ID: 11, 조문: 004800, 항: 000000, 중복 2개
-- 제목: 상관에 대한 폭행, 협박
DELETE FROM statutes_articles WHERE id IN (4965);

-- 법령ID: 11, 조문: 004900, 항: 000①00, 중복 2개
-- 제목: 상관에 대한 집단 폭행, 협박 등
DELETE FROM statutes_articles WHERE id IN (4967);

-- 법령ID: 11, 조문: 005000, 항: 000000, 중복 2개
-- 제목: 상관에 대한 특수 폭행, 협박
DELETE FROM statutes_articles WHERE id IN (4970);

-- 법령ID: 11, 조문: 005200, 항: 000②00, 중복 2개
-- 제목: 상관에 대한 폭행치사상
DELETE FROM statutes_articles WHERE id IN (4976);

-- 법령ID: 11, 조문: 005200, 항: 000000, 중복 2개
-- 제목: 상관에 대한 특수상해
DELETE FROM statutes_articles WHERE id IN (4983);

-- 법령ID: 11, 조문: 005200, 항: 000①00, 중복 2개
-- 제목: 상관에 대한 집단상해 등
DELETE FROM statutes_articles WHERE id IN (4980);

-- 법령ID: 11, 조문: 005200, 항: 000000, 중복 2개
-- 제목: 상관에 대한 상해
DELETE FROM statutes_articles WHERE id IN (4978);

-- 법령ID: 11, 조문: 005400, 항: 000000, 중복 2개
-- 제목: 초병에 대한 폭행, 협박
DELETE FROM statutes_articles WHERE id IN (4993);

-- 법령ID: 11, 조문: 005500, 항: 000①00, 중복 2개
-- 제목: 초병에 대한 집단 폭행, 협박 등
DELETE FROM statutes_articles WHERE id IN (4995);

-- 법령ID: 11, 조문: 005600, 항: 000000, 중복 2개
-- 제목: 초병에 대한 특수 폭행, 협박
DELETE FROM statutes_articles WHERE id IN (4998);

-- 법령ID: 11, 조문: 005800, 항: 000①00, 중복 2개
-- 제목: 초병에 대한 집단상해 등
DELETE FROM statutes_articles WHERE id IN (5008);

-- 법령ID: 11, 조문: 005800, 항: 000000, 중복 2개
-- 제목: 초병에 대한 중상해
DELETE FROM statutes_articles WHERE id IN (5013);

-- 법령ID: 11, 조문: 005800, 항: 000000, 중복 2개
-- 제목: 초병에 대한 상해
DELETE FROM statutes_articles WHERE id IN (5006);

-- 법령ID: 11, 조문: 005800, 항: 000②00, 중복 2개
-- 제목: 초병에 대한 폭행치사상
DELETE FROM statutes_articles WHERE id IN (5004);

-- 법령ID: 11, 조문: 005800, 항: 000000, 중복 2개
-- 제목: 초병에 대한 특수상해
DELETE FROM statutes_articles WHERE id IN (5011);

-- 법령ID: 11, 조문: 006000, 항: 000000, 중복 2개
-- 제목: 직무수행 중인 군인등에 대한 상해
DELETE FROM statutes_articles WHERE id IN (5030);

-- 법령ID: 11, 조문: 006000, 항: 000⑤00, 중복 2개
-- 제목: 직무수행 중인 군인등에 대한 폭행, 협박 등
DELETE FROM statutes_articles WHERE id IN (5028);

-- 법령ID: 11, 조문: 006000, 항: 000②00, 중복 2개
-- 제목: 직무수행 중인 군인등에 대한 폭행, 협박 등
DELETE FROM statutes_articles WHERE id IN (5022);

-- 법령ID: 11, 조문: 006000, 항: 000①00, 중복 2개
-- 제목: 직무수행 중인 군인등에 대한 폭행, 협박 등
DELETE FROM statutes_articles WHERE id IN (5020);

-- 법령ID: 11, 조문: 006000, 항: 000①00, 중복 2개
-- 제목: 직무수행 중인 군인등에 대한 집단상해 등
DELETE FROM statutes_articles WHERE id IN (5032);

-- 법령ID: 11, 조문: 006000, 항: 000000, 중복 2개
-- 제목: 직무수행 중인 군인등에 대한 중상해
DELETE FROM statutes_articles WHERE id IN (5035);

-- 법령ID: 11, 조문: 006600, 항: 000②00, 중복 2개
-- 제목: 군용시설 등에 대한 방화
DELETE FROM statutes_articles WHERE id IN (5058);

-- 법령ID: 11, 조문: 006700, 항: 000000, 중복 2개
-- 제목: 노적 군용물에 대한 방화
DELETE FROM statutes_articles WHERE id IN (5060);

-- 법령ID: 11, 조문: 007500, 항: 000①00, 중복 2개
-- 제목: 군용물 등 범죄에 대한 형의 가중
DELETE FROM statutes_articles WHERE id IN (5072);

-- 법령ID: 12, 조문: 009100, 항: 000000, 중복 2개
-- 제목: 국헌문란의 정의
DELETE FROM statutes_articles WHERE id IN (5322);

-- 법령ID: 13, 조문: 001600, 항: 000000, 중복 2개
-- 제목: 간이입소절차
DELETE FROM statutes_articles WHERE id IN (5858);

-- 법령ID: 13, 조문: 002700, 항: 000①00, 중복 2개
-- 제목: 수용자에 대한 금품 전달
DELETE FROM statutes_articles WHERE id IN (5889);

-- 법령ID: 13, 조문: 004100, 항: 000③00, 중복 2개
-- 제목: 접견
DELETE FROM statutes_articles WHERE id IN (5929);

-- 법령ID: 13, 조문: 004100, 항: 000②00, 중복 2개
-- 제목: 접견
DELETE FROM statutes_articles WHERE id IN (5927);

-- 법령ID: 13, 조문: 004500, 항: 000③00, 중복 2개
-- 제목: 종교행사의 참석 등
DELETE FROM statutes_articles WHERE id IN (5969);

-- 법령ID: 13, 조문: 004800, 항: 000②00, 중복 2개
-- 제목: 라디오 청취와 텔레비전 시청
DELETE FROM statutes_articles WHERE id IN (5977);

-- 법령ID: 13, 조문: 006300, 항: 000③00, 중복 2개
-- 제목: 교육
DELETE FROM statutes_articles WHERE id IN (6036);

-- 법령ID: 13, 조문: 007400, 항: 000①00, 중복 2개
-- 제목: 위로금ㆍ조위금
DELETE FROM statutes_articles WHERE id IN (6067);

-- 법령ID: 13, 조문: 007700, 항: 000②00, 중복 2개
-- 제목: 귀휴
DELETE FROM statutes_articles WHERE id IN (6078);

-- 법령ID: 13, 조문: 007800, 항: 000000, 중복 2개
-- 제목: 귀휴의 취소
DELETE FROM statutes_articles WHERE id IN (6082);

-- 법령ID: 13, 조문: 009500, 항: 000①00, 중복 2개
-- 제목: 보호실 수용
DELETE FROM statutes_articles WHERE id IN (6120);

-- 법령ID: 13, 조문: 009600, 항: 000①00, 중복 2개
-- 제목: 진정실 수용
DELETE FROM statutes_articles WHERE id IN (6127);

-- 법령ID: 13, 조문: 010900, 항: 000②00, 중복 2개
-- 제목: 징벌의 부과
DELETE FROM statutes_articles WHERE id IN (6222);

-- 법령ID: 13, 조문: 011000, 항: 000①00, 중복 2개
-- 제목: 징벌대상자의 조사
DELETE FROM statutes_articles WHERE id IN (6226);

-- 법령ID: 13, 조문: 012800, 항: 000③00, 중복 2개
-- 제목: 시신의 인도 등
DELETE FROM statutes_articles WHERE id IN (6300);

-- 법령ID: 13, 조문: 012800, 항: 000②00, 중복 2개
-- 제목: 시신의 인도 등
DELETE FROM statutes_articles WHERE id IN (6298);

-- 법령ID: 13, 조문: 013400, 항: 000000, 중복 2개
-- 제목: 출석의무 위반 등
DELETE FROM statutes_articles WHERE id IN (6317);

-- 법령ID: 14, 조문: 000500, 항: 000000, 중복 2개
-- 제목: 독거수용의 구분
DELETE FROM statutes_articles WHERE id IN (6337);

-- 법령ID: 14, 조문: 005900, 항: 000②00, 중복 2개
-- 제목: 변호사 와의 접견
DELETE FROM statutes_articles WHERE id IN (6447);

-- 법령ID: 14, 조문: 005900, 항: 000①00, 중복 2개
-- 제목: 변호사 와의 접견
DELETE FROM statutes_articles WHERE id IN (6445);

-- 법령ID: 14, 조문: 006200, 항: 000④00, 중복 2개
-- 제목: 접견내용의 청취·기록·녹음·녹화
DELETE FROM statutes_articles WHERE id IN (6460);

-- 법령ID: 14, 조문: 006200, 항: 000①00, 중복 2개
-- 제목: 접견내용의 청취·기록·녹음·녹화
DELETE FROM statutes_articles WHERE id IN (6456);

-- 법령ID: 16, 조문: 001400, 항: 000000, 중복 2개
-- 제목: 관할지정의 청구
DELETE FROM statutes_articles WHERE id IN (6693);

-- 법령ID: 16, 조문: 001500, 항: 000000, 중복 2개
-- 제목: 관할이전의 신청
DELETE FROM statutes_articles WHERE id IN (6695);

-- 법령ID: 16, 조문: 001800, 항: 000①00, 중복 2개
-- 제목: 기피의 원인과 신청권자
DELETE FROM statutes_articles WHERE id IN (6710);

-- 법령ID: 16, 조문: 004800, 항: 000②00, 중복 2개
-- 제목: 조서의 작성 방법
DELETE FROM statutes_articles WHERE id IN (6785);

-- 법령ID: 16, 조문: 012600, 항: 000000, 중복 2개
-- 제목: 야간집행제한의 예외
DELETE FROM statutes_articles WHERE id IN (7023);

-- 법령ID: 16, 조문: 014800, 항: 000000, 중복 2개
-- 제목: 근친자의 형사책임과 증언 거부
DELETE FROM statutes_articles WHERE id IN (7060);

-- 법령ID: 16, 조문: 015900, 항: 000000, 중복 2개
-- 제목: 선서 무능력
DELETE FROM statutes_articles WHERE id IN (7084);

-- 법령ID: 16, 조문: 019700, 항: 000①00, 중복 2개
-- 제목: 보완수사요구
DELETE FROM statutes_articles WHERE id IN (7204);

-- 법령ID: 16, 조문: 020000, 항: 000①00, 중복 2개
-- 제목: 긴급체포
DELETE FROM statutes_articles WHERE id IN (7232);

-- 법령ID: 16, 조문: 021400, 항: 000①00, 중복 2개
-- 제목: 보증금의 몰수
DELETE FROM statutes_articles WHERE id IN (7308);

-- 법령ID: 16, 조문: 021400, 항: 000⑤00, 중복 2개
-- 제목: 체포와 구속의 적부심사
DELETE FROM statutes_articles WHERE id IN (7292);

-- 법령ID: 16, 조문: 021400, 항: 000③00, 중복 2개
-- 제목: 체포와 구속의 적부심사
DELETE FROM statutes_articles WHERE id IN (7289);

-- 법령ID: 16, 조문: 021600, 항: 000①00, 중복 2개
-- 제목: 영장에 의하지 아니한 강제처분
DELETE FROM statutes_articles WHERE id IN (7313);

-- 법령ID: 16, 조문: 024400, 항: 000000, 중복 2개
-- 제목: 장애인 등 특별히 보호를 요하는 자에 대한 특칙
DELETE FROM statutes_articles WHERE id IN (7397);

-- 법령ID: 16, 조문: 024500, 항: 000000, 중복 2개
-- 제목: 사법경찰관의 사건송치 등
DELETE FROM statutes_articles WHERE id IN (7409);

-- 법령ID: 16, 조문: 026100, 항: 000000, 중복 2개
-- 제목: 지방검찰청검사장 등의 처리
DELETE FROM statutes_articles WHERE id IN (7471);

-- 법령ID: 16, 조문: 026200, 항: 000②00, 중복 2개
-- 제목: 심리와 결정
DELETE FROM statutes_articles WHERE id IN (7474);

-- 법령ID: 16, 조문: 026600, 항: 000①00, 중복 2개
-- 제목: 공판준비기일 종결의 효과
DELETE FROM statutes_articles WHERE id IN (7554);

-- 법령ID: 16, 조문: 027600, 항: 000①00, 중복 2개
-- 제목: 장애인 등 특별히 보호를 요하는 자에 대한 특칙
DELETE FROM statutes_articles WHERE id IN (7590);

-- 법령ID: 16, 조문: 037200, 항: 000000, 중복 2개
-- 제목: 비약적 상고
DELETE FROM statutes_articles WHERE id IN (7857);

-- 법령ID: 16, 조문: 043800, 항: 000②00, 중복 2개
-- 제목: 재심의 심판
DELETE FROM statutes_articles WHERE id IN (7973);

-- 법령ID: 16, 조문: 044000, 항: 000000, 중복 2개
-- 제목: 무죄판결의 공시
DELETE FROM statutes_articles WHERE id IN (7978);

-- 법령ID: 16, 조문: 044600, 항: 000000, 중복 2개
-- 제목: 파기의 판결
DELETE FROM statutes_articles WHERE id IN (7988);
