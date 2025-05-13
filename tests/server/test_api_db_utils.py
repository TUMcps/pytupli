from pytupli.schema import FilterEQ, FilterGEQ, FilterLEQ, FilterAND, FilterOR, FilterType
import pytest

from pytupli.server.db.db_handler import MongoDBHandler

convert_filter_to_query = MongoDBHandler.convert_filter_to_query


def test_convert_filter_to_query():
    # Test single base filter
    eq_filter = FilterEQ(type=FilterType.EQ, key='benchmark_id', value='123')
    assert convert_filter_to_query(eq_filter) == {'benchmark_id': '123'}

    # Test AND combination
    and_filter = FilterAND(
        type=FilterType.AND,
        filters=[
            FilterEQ(type=FilterType.EQ, key='state', value='active'),
            FilterGEQ(type=FilterType.GEQ, key='reward', value=10.0),
        ],
    )
    assert convert_filter_to_query(and_filter) == {
        '$and': [
            {'state': 'active'},
            {'reward': {'$gte': 10.0}},
        ]
    }

    # Test OR combination
    or_filter = FilterOR(
        type=FilterType.OR,
        filters=[
            FilterEQ(type=FilterType.EQ, key='status', value='pending'),
            FilterLEQ(type=FilterType.LEQ, key='time', value=30),
        ],
    )
    assert convert_filter_to_query(or_filter) == {
        '$or': [
            {'status': 'pending'},
            {'time': {'$lte': 30}},
        ]
    }

    # Test nested combination
    nested_filter = FilterAND(
        type=FilterType.AND,
        filters=[
            FilterEQ(type=FilterType.EQ, key='state', value='active'),
            FilterOR(
                type=FilterType.OR,
                filters=[
                    FilterGEQ(type=FilterType.GEQ, key='reward', value=5.0),
                    FilterLEQ(type=FilterType.LEQ, key='time', value=100),
                ],
            ),
        ],
    )
    assert convert_filter_to_query(nested_filter) == {
        '$and': [
            {'state': 'active'},
            {
                '$or': [
                    {'reward': {'$gte': 5.0}},
                    {'time': {'$lte': 100}},
                ]
            },
        ]
    }


def test_complex_filter_chain():
    complex_filter = FilterAND(
        type=FilterType.AND,
        filters=[
            FilterOR(
                type=FilterType.OR,
                filters=[
                    FilterEQ(type=FilterType.EQ, key='state', value='active'),
                    FilterGEQ(type=FilterType.GEQ, key='score', value=90),
                ],
            ),
            FilterAND(
                type=FilterType.AND,
                filters=[
                    FilterEQ(type=FilterType.EQ, key='validated', value='true'),
                    FilterOR(
                        type=FilterType.OR,
                        filters=[
                            FilterLEQ(type=FilterType.LEQ, key='time', value=50),
                            FilterGEQ(type=FilterType.GEQ, key='reward', value=7.5),
                        ],
                    ),
                ],
            ),
        ],
    )

    expected_query = {
        '$and': [
            {
                '$or': [
                    {'state': 'active'},
                    {'score': {'$gte': 90}},
                ]
            },
            {
                '$and': [
                    {'validated': 'true'},
                    {
                        '$or': [
                            {'time': {'$lte': 50}},
                            {'reward': {'$gte': 7.5}},
                        ]
                    },
                ]
            },
        ]
    }

    assert convert_filter_to_query(complex_filter) == expected_query


if __name__ == '__main__':
    pytest.main(['-v', __file__])
